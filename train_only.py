
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as T
import custom_augmentations as CA
import test
import util
import our_parser
import commons
import cosface_loss
import new_cosface_loss
import sphereface_loss
import arcface_loss
import augmentations
from model import network
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset
from datasets.grl_datasets import GrlDataset
import random 

torch.backends.cudnn.benchmark = True  # Provides a speedup

args = our_parser.parse_arguments()
start_time = datetime.now()
output_folder = f"AML23-CosPlace/model/results/best_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(output_folder, console="debug")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {output_folder}")

#### Model (+ GRL for domain adaptation)
model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim, grl = args.grl)

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

if args.resume_model is not None:
    logging.debug(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    model.load_state_dict(model_state_dict)

model = model.to(args.device).train()

#### Optimizer
criterion = torch.nn.CrossEntropyLoss()
epoch_grl_loss = 0 
domain_adapt_criterion = torch.nn.CrossEntropyLoss() if args.grl == True else None #Loss for Domain Adaptation
model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#### Datasets
groups = [TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                       current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]
grl_dataset = GrlDataset(sf_xs_train_path="/content/small/train", target_path="/content/AML23-CosPlace/our_FDA/target") if args.grl == True else None #To Do... Add Target Dataset for Domain Adaptation

# Each group has its own classifier, which depends on the number of classes in the group
# Each group has its own classifier, which depends on the number of classes in the group
classifiers = None
if args.loss_function == "cosface":
    classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]
elif args.loss_function =="arcface":
    classifiers = [arcface_loss.ArcFace(args.fc_output_dim, len(group)) for group in groups]
elif args.loss_function =="sphereface":
    classifiers = [sphereface_loss.SphereFace(args.fc_output_dim, len(group)) for group in groups]
elif args.loss_function =="new_cosface":
    classifiers = [new_cosface_loss.CosFace(args.fc_output_dim, len(group)) for group in groups]

classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers]

logging.info(f"Using {len(groups)} groups")
logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

#### Resume
if args.resume_train:
    model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num = \
        util.resume_train(args, output_folder, model, model_optimizer, classifiers, classifiers_optimizers)
    model = model.to(args.device)
    epoch_num = start_epoch_num - 1
    logging.info(f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} from checkpoint {args.resume_train}")
else:
    best_val_recall1 = start_epoch_num = 0

#### Train / evaluation loop
logging.info("Start training ...")
logging.info(f"There are {len(groups[0])} classes for the first group, " +
             f"each epoch has {args.iterations_per_epoch} iterations " +
             f"with batch_size {args.batch_size}, therefore the model sees each class (on average) " +
             f"{args.iterations_per_epoch * args.batch_size / len(groups[0]):.1f} times per epoch")

val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold)
logging.info(f"Validation set: {val_ds}")



if args.augmentation_device == "cuda":
    random.seed(4321)
    gpu_augmentation = T.Compose([
            augmentations.DeviceAgnosticColorJitter(brightness=args.brightness,
                                                    contrast=args.contrast,
                                                    saturation=args.saturation,
                                                    hue=args.hue),
            augmentations.DeviceAgnosticRandomResizedCrop([512, 512],
                                                          scale=[1-args.random_resized_crop, 1]),
                                                          T.RandomHorizontalFlip(p=args.hflip),
                                                          CA.RandomGaussianBlur(p=args.gblur,kernel_size=1,sigma=(0.1,2.0)),
                                                          T.RandomGrayscale(p=args.rgrayscale),
                                                          T.RandomErasing(p=args.rerasing, scale=(0.02, 0.25), ratio=(0.3, 3.3), value=0, inplace=False),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

if args.use_amp16:
    scaler = torch.cuda.amp.GradScaler()

for epoch_num in range(start_epoch_num, args.epochs_num):
    
    #### Train
    epoch_start_time = datetime.now()
    # Select classifier and dataloader according to epoch
    current_group_num = epoch_num % args.groups_num
    classifiers[current_group_num] = classifiers[current_group_num].to(args.device)
    util.move_to_device(classifiers_optimizers[current_group_num], args.device)
    
    dataloader = commons.InfiniteDataLoader(groups[current_group_num], num_workers=args.num_workers,
                                            batch_size=args.batch_size, shuffle=True,
                                            pin_memory=(args.device == "cuda"), drop_last=True)

    domain_adapt_dataloader = torch.utils.data.DataLoader(grl_dataset, num_workers=args.num_workers,
                                            batch_size=args.batch_size, shuffle=True,
                                            pin_memory=(args.device == "cuda"), drop_last=True) if args.grl == True else None
    
    dataloader_iterator = iter(dataloader)
    domain_adapt_dataloader_iterator = iter(domain_adapt_dataloader) if args.grl == True else None
    model = model.train()
    
    epoch_losses = np.zeros((0, 1), dtype=np.float32)
    for iteration in tqdm(range(args.iterations_per_epoch), ncols=100):
        images, targets, _ = next(dataloader_iterator)
        images, targets = images.to(args.device), targets.to(args.device)

        if (args.grl == True):
            domain_adapt_images, domain_adapt_labels = next(domain_adapt_dataloader_iterator)
            domain_adapt_images, domain_adapt_labels = domain_adapt_images.to(args.device), domain_adapt_labels.to(args.device)
        
        if args.augmentation_device == "cuda":
            images = gpu_augmentation(images)
        
        model_optimizer.zero_grad()
        classifiers_optimizers[current_group_num].zero_grad()
        
        if not args.use_amp16:
            descriptors = model(images) #feature extraction -> theta_f
            output = classifiers[current_group_num](descriptors, targets) #output probabilities for each class -> theta_y
            loss = criterion(output, targets) #scalar value that represents the difference between the predicted and true class probabilities.
            loss.backward()
            '''
            Domain Adaptation here
            '''
            domain_adapt_loss = 0 #Initialized to 0 so that i can append it anyway to epoch_losses
            if (args.grl == True):
                alpha = 0.1 #GRL trade-off value
                domain_adapt_output = model(domain_adapt_images, grl=args.grl)
                domain_adapt_loss = domain_adapt_criterion(domain_adapt_output, domain_adapt_labels)
                domain_adapt_loss = domain_adapt_loss * alpha
                domain_adapt_loss.backward()
                domain_adapt_loss = domain_adapt_loss.item() #.item() returns tensor value as a standard number
                epoch_grl_loss += domain_adapt_loss
                del domain_adapt_images, domain_adapt_output

            epoch_losses = np.append(epoch_losses, loss.item() + domain_adapt_loss) #append loss (L_f + alpha*L_CE)
            del loss, domain_adapt_loss, output, images
            model_optimizer.step() #optimize parameters
            classifiers_optimizers[current_group_num].step() 
        else:  # Use AMP 16
            with torch.cuda.amp.autocast():
                descriptors = model(images)
                output = classifiers[current_group_num](descriptors, targets)
                loss = criterion(output, targets)
                domain_adapt_loss = 0
                if (args.grl == True):
                    alpha = 0.1
                    domain_adapt_output = model(domain_adapt_images, grl=args.grl)
                    domain_adapt_loss = domain_adapt_criterion(domain_adapt_output, domain_adapt_labels)
                    domain_adapt_loss = domain_adapt_loss * alpha
                    epoch_grl_loss += domain_adapt_loss.item()
                    del domain_adapt_images, domain_adapt_output
            scaler.scale(loss + domain_adapt_loss).backward()
            epoch_losses = np.append(epoch_losses, loss.item() + domain_adapt_loss.item())
            del loss, domain_adapt_loss, output, images
            scaler.step(model_optimizer)
            scaler.step(classifiers_optimizers[current_group_num])
            scaler.update()
    
    classifiers[current_group_num] = classifiers[current_group_num].cpu()
    util.move_to_device(classifiers_optimizers[current_group_num], "cpu")
    
    logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                  f"loss = {epoch_losses.mean():.4f}")
    if args.grl: logging.debug(f"Average GRL epoch loss (* alpha = 0.1): {epoch_grl_loss/args.iterations_per_epoch:.4f}")
    
    #### Evaluation
    recalls, recalls_str = test.test(args, val_ds, model)
    logging.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, {val_ds}: {recalls_str[:20]}")
    is_best = recalls[0] > best_val_recall1
    best_val_recall1 = max(recalls[0], best_val_recall1)
    # Save checkpoint, which contains all training parameters
    util.save_checkpoint({
        "epoch_num": epoch_num + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model_optimizer.state_dict(),
        "classifiers_state_dict": [c.state_dict() for c in classifiers],
        "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
        "best_val_recall1": best_val_recall1
    }, is_best, output_folder)


logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Best model is saved in f"{output_folder}/best_model.pth"
logging.info(f"Best model is saved in {output_folder}/best_model.pth")
logging.info("Experiment finished (without any errors)")
