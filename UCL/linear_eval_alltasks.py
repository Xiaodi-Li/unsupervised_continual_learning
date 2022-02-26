import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model, get_backbone
from tools import AverageMeter, knn_monitor
from datasets import get_dataset
from models.optimizers import get_optimizer, LR_Scheduler
from utils.loggers import *

def evaluate(model, classifier, dataset, device, last=False) -> Tuple[list, list, list, list]:
    classifier.eval()
    # acc_meter.reset()
    accs, accs_mask_classes = [], []
    knn_accs, knn_accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct = correct_mask_classes = total = 0
        for images, labels in test_loader:
          with torch.no_grad():
            feature = model(images.to(args.device))
            outputs = classifier(feature)
            _, preds = torch.max(outputs.data, 1)
            correct += (preds == labels.to(args.device)).sum().item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                outputs = mask_classes(outputs, dataset, k)
                _, preds = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(preds == labels.to(device)).item()

        knn_acc, knn_acc_mask = knn_monitor(model.net.module.backbone, dataset, dataset.memory_loaders[k], dataset.test_loaders[k], device, args.cl_default, task_id=k, k=min(args.train.knn_k, len(dataset.memory_loaders[k].dataset))) 

        knn_accs.append(knn_acc
                    if 'class-il' in model.COMPATIBILITY else 0)
        knn_accs_mask_classes.append(knn_acc_mask)
         
        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
    return accs, accs_mask_classes, knn_accs, knn_accs_mask_classes

def evaluate_single(model, classifier, dataset, test_loader, memory_loader, device, k, last=False) -> Tuple[list, list, list, list]:
    classifier.eval()
    # acc_meter.reset()
    accs, accs_mask_classes = [], []
    knn_accs, knn_accs_mask_classes = [], []
    correct = correct_mask_classes = total = 0
    # for images, labels in test_loader:
        # with torch.no_grad():
          # feature = model.net.module.backbone(images.to(args.device), return_features=True)
          # outputs = classifier(feature)
          # _, preds = torch.max(outputs.data, 1)
          # correct += (preds == labels.to(args.device)).sum().item()
          # total += labels.shape[0]

    knn_acc, knn_acc_mask = knn_monitor(model.net.module.backbone, dataset, memory_loader, test_loader, device, args.cl_default, task_id=k, k=min(args.train.knn_k, len(dataset.memory_loaders[k].dataset))) 

    # return correct/total * 100, knn_acc
    return knn_acc



def main(device, args):

    dataset_copy = get_dataset(args)
    train_loader, memory_loader, test_loader = dataset_copy.get_data_loaders(args)
    model = get_model(args, device, len(train_loader), get_aug(train=False, train_classifier=False, **args.aug_kwargs))
    classifier = nn.Linear(in_features=model.net.module.backbone.output_dim, out_features=100, bias=True).to(args.device)
    classifier = torch.nn.DataParallel(classifier)

    for t in range(dataset_copy.N_TASKS - 1):
        _, _, _ = dataset_copy.get_data_loaders(args)

    # Start training
    linear_acc, knn_acc = [], []
    for t in tqdm(range(0, dataset_copy.N_TASKS), desc='Evaluatinng'):
      dataset = get_dataset(args)
      model_path = os.path.join(args.ckpt_dir, args.name + '.pth')
      save_dict = torch.load(model_path, map_location='cpu')

      msg = model.net.module.backbone.load_state_dict({k[16:]:v for k, v in save_dict['state_dict'].items() if 'backbone.' in k}, strict=True)
      model = model.to(args.device)
      classifier = nn.Linear(in_features=model.net.module.backbone.output_dim, out_features=100, bias=True).to(args.device)
    
      optimizer = get_optimizer(
        args.eval.optimizer.name, classifier, 
        lr=args.eval.base_lr*args.eval.batch_size/256, 
        momentum=args.eval.optimizer.momentum, 
        weight_decay=args.eval.optimizer.weight_decay)

      # define lr scheduler
      lr_scheduler = LR_Scheduler(
        optimizer,
        args.eval.warmup_epochs, args.eval.warmup_lr*args.eval.batch_size/256, 
        args.eval.num_epochs, args.eval.base_lr*args.eval.batch_size/256, args.eval.final_lr*args.eval.batch_size/256, 
        len(train_loader),
      )
      task_linear_acc, task_knn_acc = [], []
      for t1 in tqdm(range(0, dataset_copy.N_TASKS), desc='Inner tasks'):
        train_loader, memory_loader, test_loader = dataset.get_data_loaders(args)
        t1_knn_acc = evaluate_single(model, classifier, dataset, test_loader, memory_loader, device, t1)
        task_knn_acc.append(t1_knn_acc)
      knn_acc.append(task_knn_acc)
      print(f'Task {t}: {task_knn_acc}')
    
    mean_knn_acc = sum(knn_acc[-1][:len(knn_acc[-1])]) / len(knn_acc[-1])
    print(f'KNN accuracy on Task {t1}: {mean_knn_acc}')

    max_knn_acc = [max(idx) for idx in zip(*knn_acc)]
    mean_knn_fgt = sum([x1 - x2 for (x1, x2) in zip(max_knn_acc, knn_acc[-1])]) / len(knn_acc[-1])
    print(f'KNN Forgetting: {mean_knn_fgt}')


if __name__ == "__main__":
    args = get_args()
    main(device=args.device, args=args)
