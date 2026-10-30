# Consignes 3, 4 et 6 : comparer les représentations des tâches prétextes
# sur CIFAR-10 et STL-10, avec un classifieur linéaire sur l'encodeur figé.
# Code des cellules « Function for evaluation and comparison » du notebook ;
# à exécuter dans le dossier cs294-158-ssl (après data.zip, results.zip et
# l'entraînement de src/position_relative.py, qui produit position_relative.pth).

import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as data
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from deepul_helper.data import get_transform
from deepul_helper.demos import load_model_and_data
from deepul_helper.utils import accuracy

from position_relative import RelativePosition


class LocalSTL10(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform

        split = "train" if train else "test"
        self.dataset = load_dataset(
            "tanganke/stl10",
            split=split
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        img = item["image"].convert("RGB")
        target = int(item["label"])

        if self.transform is not None:
            img = self.transform(img)

        return img, target


# Enum useful datasets for the comparison
DATASETS = {
    'cifar10': (
        lambda racine, train, tf: datasets.CIFAR10(
            racine, train=train, transform=tf, download=True
        ),
        10
    ),

    'stl10': (
      lambda racine, train, tf: LocalSTL10(
          racine, train=train, transform=tf
      ),
      10
    ),

    # 'cifar100': (
    #     lambda racine, train, tf: datasets.CIFAR100(
    #         racine, train=train, transform=tf, download=True
    #     ),
    #     100
    # )
}

DEMOS = ['context_encoder', 'rotation', 'simclr', 'relative_position']

# La position relative n'a pas de prétraitement propre dans la librairie :
# elle a été entraînée avec celui de la rotation, on garde donc le même.
PRETRAITEMENT = {'relative_position': 'rotation'}


def charger_dataset(task, dataset):
    """Loaders du dataset, préparés comme les images de test CIFAR-10 de la démo."""
    construire, n_classes = DATASETS[dataset]
    tf = transforms.Compose([transforms.Resize((32, 32)),
                             get_transform('cifar10', PRETRAITEMENT.get(task, task), train=False)])
    racine = osp.join('data', dataset)
    train_set = construire(racine, True, tf)
    test_set = construire(racine, False, tf)
    train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)
    return train_loader, test_loader, n_classes


def charger_encodeur(task):
    """Encodeur pré-entraîné sur CIFAR-10 (+ classifieur du checkpoint s'il existe)."""
    if task == 'relative_position':
        model = RelativePosition().cuda()
        model.load_state_dict(torch.load('position_relative.pth', map_location='cuda'))
        model.eval()
        return model, None  # pas de classifieur appris pendant le pré-entraînement
    model, classifieur, _, _ = load_model_and_data(task)
    return model, classifieur


def representation(model, task, images):
    """Image -> vecteur appris par l'encodeur gelé (ce que le classifieur linéaire reçoit)."""
    if task == 'simclr':
        images = images[0]  # SimCLR fournit 2 vues de chaque image ; une seule suffit
    return model.encode(images.cuda())


def score(model, task, classifieur, loader, n_max=None):
    """Top-1 et top-5 (en %) du classifieur sur le loader (ou ses n_max premières images)."""
    classifieur.eval()
    bons1 = bons5 = total = 0
    with torch.no_grad():
        for images, y in loader:
            logits = classifieur(representation(model, task, images))
            top1, top5 = accuracy(logits, y.cuda(), topk=(1, 5))
            bons1 += top1.item() * len(y)
            bons5 += top5.item() * len(y)
            total += len(y)
            if n_max and total >= n_max:
                break
    return bons1 / total, bons5 / total


def entrainer_lineaire(model, task, loader, n_classes, n_train=10000, epochs=30):
    """Entraîne un nouveau classifieur linéaire sur les représentations de l'encodeur gelé."""
    # 1. Passer n_train images dans l'encodeur, une seule fois
    zs, ys = [], []
    with torch.no_grad():
        for images, y in loader:
            zs.append(representation(model, task, images))
            ys.append(y.cuda())
            if sum(len(b) for b in ys) >= n_train:
                break
    z, y = torch.cat(zs)[:n_train], torch.cat(ys)[:n_train]

    # 2. Même architecture que le classifieur de la démo, avec le bon nombre de classes
    model.n_classes = n_classes
    classifieur = model.construct_classifier().cuda()
    opt = torch.optim.Adam(classifieur.parameters(), lr=1e-3)

    # 3. Descente de gradient classique ; seul le classifieur apprend
    classifieur.train()
    for _ in range(epochs):
        ordre = torch.randperm(len(y), device=y.device)
        for k in range(0, len(y), 256):
            idx = ordre[k:k + 256]
            if len(idx) < 2:  # BatchNorm refuse un lot d'une seule image
                continue
            perte = F.cross_entropy(classifieur(z[idx]), y[idx])
            opt.zero_grad()
            perte.backward()
            opt.step()
    return classifieur


def evaluer_demo(task, dataset='cifar10', reentrainer=None, n_train=10000, epochs=30):
    """Évalue la représentation d'une méthode sur un dataset ; renvoie une ligne du tableau."""
    if reentrainer is None:
        # La position relative n'a pas de classifieur de checkpoint : on le réentraîne toujours
        reentrainer = dataset != 'cifar10' or task == 'relative_position'
    if not reentrainer and (dataset != 'cifar10' or task == 'relative_position'):
        raise ValueError("Pas de classifieur de checkpoint utilisable : mettre reentrainer=True.")

    model, classifieur = charger_encodeur(task)
    train_loader, test_loader, n_classes = charger_dataset(task, dataset)

    if reentrainer:
        classifieur = entrainer_lineaire(model, task, train_loader, n_classes, n_train, epochs)
        train1, train5 = score(model, task, classifieur, train_loader, n_max=n_train)
    else:
        train1, train5 = score(model, task, classifieur, train_loader)
    test1, test5 = score(model, task, classifieur, test_loader)

    print(f'{task:17s} {dataset:9s} test top-1 = {test1:.1f} %')
    return {'démo': task, 'dataset': dataset,
            'classifieur': 'réentraîné' if reentrainer else 'checkpoint',
            'train top-1': train1, 'train top-5': train5,
            'test top-1': test1, 'test top-5': test5}


def tracer(tableau, colonne='test top-1'):
    """Barres groupées : une barre par dataset, un groupe par démo."""
    pivot = tableau.pivot_table(index='démo', columns='dataset', values=colonne).loc[DEMOS]
    ax = pivot.plot.bar(rot=0, figsize=(8, 4))
    for barres in ax.containers:
        ax.bar_label(barres, fmt='%.1f')
    ax.set_ylim(0, 100)
    ax.set_ylabel(f'{colonne} (%)')
    ax.set_title('Qualité de la représentation apprise par chaque tâche prétexte')
    plt.show()


if __name__ == '__main__':
    lignes = [evaluer_demo(t, ds, reentrainer=True) for t in DEMOS for ds in ['cifar10', 'stl10']]
    tableau = pd.DataFrame(lignes)
    print(tableau.round(1))
    tableau.round(1).to_csv('comparaison.csv', index=False)
    tracer(tableau)
    plt.savefig('comparaison.png', dpi=150, bbox_inches='tight')
