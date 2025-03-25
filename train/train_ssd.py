import argparse
import os
import logging
import sys
import itertools

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
# from vision.ssd.vgg_ssd import create_vgg_ssd
# from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
# from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
# from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.datasets.slr_dataset import SLRDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.nn.multibox_loss import MultiboxLoss
# from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
# from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

from vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite
from config import default_config as cfg  # Import the default config instance



logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")


def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    print(f"Starting training for epoch {epoch}...")
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    
    progress_bar = tqdm(total=len(loader), desc=f"Epoch {epoch}", 
                        bar_format='{l_bar}{bar:30}{r_bar}')
    
    for i, data in enumerate(loader):
        try:
            images, boxes, labels = data
            if i % debug_steps == 0:
                print(f"Batch {i}: images shape {images.shape}, boxes shape {boxes.shape}, labels shape {labels.shape}")
            
            images = images.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_regression_loss += regression_loss.item()
            running_classification_loss += classification_loss.item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'reg_loss': f'{regression_loss.item():.4f}',
                'cls_loss': f'{classification_loss.item():.4f}'
            })
            progress_bar.update(1)
            
            if i and i % debug_steps == 0:
                avg_loss = running_loss / debug_steps
                avg_reg_loss = running_regression_loss / debug_steps
                avg_clf_loss = running_classification_loss / debug_steps
                logging.info(
                    f"Epoch: {epoch}, Step: {i}, " +
                    f"Average Loss: {avg_loss:.4f}, " +
                    f"Average Regression Loss {avg_reg_loss:.4f}, " +
                    f"Average Classification Loss: {avg_clf_loss:.4f}"
                )
                running_loss = 0.0
                running_regression_loss = 0.0
                running_classification_loss = 0.0
        except Exception as e:
            print(f"Error in batch {i}: {str(e)}")
            raise e
    
    progress_bar.close()
    print(f"Completed training for epoch {epoch}")
    return running_loss / len(loader)


def test(loader, net, criterion, device):
    print("Starting validation...")
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    
    progress_bar = tqdm(total=len(loader), desc="Validation", 
                        bar_format='{l_bar}{bar:30}{r_bar}')
    
    for i, data in enumerate(loader):
        try:
            images, boxes, labels = data
            if i % 10 == 0:
                print(f"Validation batch {i}: Processing {images.shape[0]} images")
                
            images = images.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)
            num += 1

            with torch.no_grad():
                confidence, locations = net(images)
                regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
                loss = regression_loss + classification_loss

            running_loss += loss.item()
            running_regression_loss += regression_loss.item()
            running_classification_loss += classification_loss.item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'reg_loss': f'{regression_loss.item():.4f}',
                'cls_loss': f'{classification_loss.item():.4f}'
            })
            progress_bar.update(1)
            
        except Exception as e:
            print(f"Error in validation batch {i}: {str(e)}")
            raise e
    
    progress_bar.close()
    avg_loss = running_loss / num
    print(f"Validation complete. Average loss: {avg_loss:.4f}")
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':
    timer = Timer()
    
    print("=== Starting SSD Model Training ===")
    print(f"Configuration: {cfg}")

    # Chuẩn bị thư mục lưu mô hình
    base_checkpoint_folder = cfg.checkpoint_folder
    if not os.path.exists(base_checkpoint_folder):
        os.makedirs(base_checkpoint_folder)
    
    # Tạo thư mục trainN cho lần train mới
    existing_train_dirs = [d for d in os.listdir(base_checkpoint_folder) 
                          if os.path.isdir(os.path.join(base_checkpoint_folder, d)) 
                          and d.startswith('train')]
    train_numbers = [int(d.replace('train', '')) for d in existing_train_dirs if d[5:].isdigit()]
    next_train_number = 1 if not train_numbers else max(train_numbers) + 1
    train_dir = f"train{next_train_number}"
    
    checkpoint_folder = os.path.join(base_checkpoint_folder, train_dir)
    os.makedirs(checkpoint_folder, exist_ok=True)
    print(f"Created checkpoint directory: {checkpoint_folder}")
    
    # Tạo file CSV để lưu thông số huấn luyện
    csv_file = os.path.join(checkpoint_folder, "training_log.csv")
    with open(csv_file, 'w') as f:
        f.write("epoch,train_loss,val_loss,val_regression_loss,val_classification_loss,learning_rate\n")
    
    logging.info(cfg)
    if cfg.net == 'mb2-ssd-lite':
        create_net = lambda num: create_mobilenetv2_ssd_ssd_lite(num, width_mult=cfg.mb2_width_mult)
        config = mobilenetv1_ssd_config
    elif cfg.net == 'mb3-ssd-lite':
        create_net = lambda num: create_mobilenetv3_ssd_lite(num)
        config = mobilenetv1_ssd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    logging.info("Prepare training datasets.")
    datasets = []
    
    print("\n=== Preparing datasets ===")
    
    for dataset_path in cfg.datasets:
        print(f"Loading dataset from: {dataset_path}")
        if cfg.dataset_type == 'slr':
            dataset = SLRDataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform)
            label_file = os.path.join(checkpoint_folder, "slr-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
            print(f"Loaded SLR dataset with {len(dataset.class_names)} classes")
        elif cfg.dataset_type == 'open_images':
            dataset = OpenImagesDataset(dataset_path,
                 transform=train_transform, target_transform=target_transform,
                 dataset_type="train", balance_data=cfg.balance_data)
            label_file = os.path.join(checkpoint_folder, "open-images-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            logging.info(dataset)
            num_classes = len(dataset.class_names)
            print(f"Loaded Open Images dataset with {len(dataset.class_names)} classes")

        else:
            raise ValueError(f"Dataset tpye {cfg.dataset_type} is not supported.")
        datasets.append(dataset)
    logging.info(f"Stored labels into file {label_file}.")
    train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, cfg.batch_size,
                              num_workers=cfg.num_workers,
                              shuffle=True,
                              drop_last=True)
    logging.info("Prepare Validation datasets.")
    if cfg.dataset_type == "slr":
        val_dataset = SLRDataset(cfg.validation_dataset, transform=test_transform,
                                 target_transform=target_transform, is_test=True)
    elif cfg.dataset_type == 'open_images':
        val_dataset = OpenImagesDataset(dataset_path,
                                        transform=test_transform, target_transform=target_transform,
                                        dataset_type="test")
        logging.info(val_dataset)
    logging.info("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, cfg.batch_size,
                            num_workers=cfg.num_workers,
                            shuffle=False,
                            drop_last=True)
    logging.info("Build network.")
    print("\n=== Building network ===")
    net = create_net(num_classes)
    print(f"Created network with {num_classes} classes using {cfg.net} architecture")

  


    min_loss = float('inf')  # Khởi tạo với giá trị vô cùng để so sánh
    last_epoch = -1
    best_model_path = os.path.join(checkpoint_folder, f"{cfg.net}-best.pth")

    base_net_lr = cfg.base_net_lr if cfg.base_net_lr is not None else cfg.lr
    extra_layers_lr = cfg.extra_layers_lr if cfg.extra_layers_lr is not None else cfg.lr
    if cfg.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif cfg.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    timer.start("Load Model")
    print("\n=== Loading model ===")
    if cfg.resume:
        print(f"Resuming from checkpoint: {cfg.resume}")
        logging.info(f"Resume from the model {cfg.resume}")
        net.load(cfg.resume)
    elif cfg.base_net:
        print(f"Initializing from base network: {cfg.base_net}")
        logging.info(f"Init from base net {cfg.base_net}")
        net.init_from_base_net(cfg.base_net)
    elif cfg.pretrained_ssd:
        print(f"Initializing from pretrained SSD: {cfg.pretrained_ssd}")
        logging.info(f"Init from pretrained ssd {cfg.pretrained_ssd}")
        net.init_from_pretrained_ssd(cfg.pretrained_ssd)
    print(f'Model loaded successfully in {timer.end("Load Model"):.2f} seconds')

    net.to(DEVICE)

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)
    logging.info(f"Learning rate: {cfg.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    if cfg.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in cfg.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                                     gamma=0.1, last_epoch=last_epoch)
    elif cfg.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, cfg.t_max, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {cfg.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    print("\n=== Setting up optimizer and scheduler ===")
    
    print(f"\n=== Starting training for {cfg.num_epochs} epochs ===")
    print(f"Training will start from epoch {last_epoch + 1}")

    #sys.exit(0)#test

    for epoch in range(last_epoch + 1, cfg.num_epochs):
        print(f"\n=== Epoch {epoch}/{cfg.num_epochs-1} ===")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        
        epoch_progress = tqdm(total=cfg.num_epochs, initial=epoch, 
                             desc="Total training progress", position=0, 
                             bar_format='{l_bar}{bar:30}{r_bar}')
                             
        # Train epoch
        running_loss = train(train_loader, net, criterion, optimizer,
              device=DEVICE, debug_steps=cfg.debug_steps, epoch=epoch)
        
        # Đánh giá mô hình sau mỗi epoch để lưu thông số
        val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
        logging.info(
            f"Epoch: {epoch}, " +
            f"Validation Loss: {val_loss:.4f}, " +
            f"Validation Regression Loss {val_regression_loss:.4f}, " +
            f"Validation Classification Loss: {val_classification_loss:.4f}"
        )
        
        # Lưu thông số vào file CSV
        with open(csv_file, 'a') as f:
            f.write(f"{epoch},{running_loss:.6f},{val_loss:.6f},{val_regression_loss:.6f},"
                    f"{val_classification_loss:.6f},{optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Lưu model sau mỗi 5 epoch
        if epoch % 5 == 0 or epoch == cfg.num_epochs - 1:
            model_path = os.path.join(checkpoint_folder, f"{cfg.net}-Epoch-{epoch}.pth")
            print(f"Saving model to: {model_path}")
            net.save(model_path)
            print(f"Model saved successfully")
            logging.info(f"Saved model {model_path}")
        
        # Cập nhật mô hình tốt nhất
        if val_loss < min_loss:
            min_loss = val_loss
            print(f"New best model with validation loss: {val_loss:.4f}")
            net.save(best_model_path)
            logging.info(f"Saved best model to {best_model_path}")
        
        scheduler.step()
        print(f"Learning rate updated. New LR: {optimizer.param_groups[0]['lr']}")
        
        epoch_progress.update(1)
        epoch_progress.set_postfix({'val_loss': f'{val_loss:.4f}', 'best_loss': f'{min_loss:.4f}'})
    
    print("\n=== Training completed successfully ===")
    print(f"Training logs saved to: {csv_file}")
    print(f"Best model saved to: {best_model_path}")
