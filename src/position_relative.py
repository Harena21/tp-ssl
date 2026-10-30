# Consigne 5 : tâche prétexte « position relative » (Doersch et al., 2015), sur CIFAR-10.
# Code de la cellule « Patch method » du notebook ; à exécuter dans le dossier cs294-158-ssl.
# On réutilise la librairie deepul_helper : l'encodeur NetworkInNetwork de la rotation,
# le prétraitement de get_transform et la même interface (forward, encode, construct_classifier).

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import datasets

from deepul_helper.data import get_transform
from deepul_helper.tasks.rotation import NetworkInNetwork

# Les 8 voisins du patch central, numérotés comme la classe à prédire :
#   0 1 2
#   3 . 4
#   5 6 7
VOISINS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]


class RelativePosition(nn.Module):
    metrics = ['Loss', 'Acc1']
    metrics_fmt = [':.4e', ':6.2f']

    def __init__(self, dataset='cifar10', n_classes=10, cellule=10, patch=8, jitter=1, p_gris=0.66):
        super().__init__()
        # Même encodeur et même couche que la rotation : la comparaison sera équitable
        self.model = NetworkInNetwork()
        self.feat_layer = 'conv2'
        self.latent_dim = 192 * 8 * 8  # conv2 d'une image 32 x 32 entière
        self.dataset, self.n_classes = dataset, n_classes
        # Grille 3 x 3 de cellules ; dans chaque cellule, un patch plus petit
        # -> un écart entre patchs voisins, et un décalage aléatoire (jitter)
        self.cellule, self.patch, self.jitter, self.p_gris = cellule, patch, jitter, p_gris

        with torch.no_grad():  # taille de conv2 pour un patch (192 x 2 x 2 pour 8 x 8)
            d_patch = self._features(torch.zeros(2, 3, patch, patch)).shape[1]
        # Tête de la tâche prétexte : les 2 patchs côte à côte -> 8 positions possibles
        self.tete = nn.Sequential(
            nn.Linear(2 * d_patch, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, 8))

    def construct_classifier(self):
        # Le même classifieur linéaire que la rotation sur CIFAR-10
        return nn.Sequential(
            nn.BatchNorm1d(self.latent_dim, affine=False),
            nn.Linear(self.latent_dim, self.n_classes))

    def _features(self, x):
        return self.model(x, out_feat_keys=(self.feat_layer,)).flatten(start_dim=1)

    def forward(self, images):
        centre, voisin, cibles = self._paires(images)
        f = self._features(torch.cat((centre, voisin)))  # encodeur partagé (siamois)
        f_centre, f_voisin = f.chunk(2)
        logits = self.tete(torch.cat((f_centre, f_voisin), dim=1))
        loss = F.cross_entropy(logits, cibles)
        acc = (logits.argmax(dim=-1) == cibles).float().mean() * 100.
        return dict(Loss=loss, Acc1=acc), None  # la représentation s'obtient avec encode()

    def encode(self, images):
        # Après l'entraînement, on encode l'image ENTIÈRE, comme pour la rotation
        return self._features(images)

    def _paires(self, images):
        """Découpe 9 patchs par image ; renvoie 8 paires (centre, voisin) et leurs classes."""
        B, C, H, W = images.shape
        P, J, cel = self.patch, self.jitter, self.cellule
        dev = images.device

        # Position de la grille dans l'image (30 px dans 32 px : 0 à 2 px de marge)
        oy = torch.randint(0, H - 3 * cel + 1, (B, 1), device=dev)
        ox = torch.randint(0, W - 3 * cel + 1, (B, 1), device=dev)
        lig = torch.arange(3, device=dev).repeat_interleave(3)  # 0 0 0 1 1 1 2 2 2
        col = torch.arange(3, device=dev).repeat(3)             # 0 1 2 0 1 2 0 1 2
        marge = (cel - P) // 2
        y = oy + lig * cel + marge + torch.randint(-J, J + 1, (B, 9), device=dev)
        x = ox + col * cel + marge + torch.randint(-J, J + 1, (B, 9), device=dev)
        y, x = y.clamp(0, H - P), x.clamp(0, W - P)

        # Extraction des 9 patchs P x P de chaque image en une seule opération
        d = torch.arange(P, device=dev)
        lignes = (y[..., None] + d)[..., :, None]   # (B, 9, P, 1)
        colonnes = (x[..., None] + d)[..., None, :]  # (B, 9, 1, P)
        b = torch.arange(B, device=dev)[:, None, None, None]
        patchs = images.permute(0, 2, 3, 1)[b, lignes, colonnes]  # (B, 9, P, P, C)
        patchs = patchs.permute(0, 1, 4, 2, 3).contiguous()        # (B, 9, C, P, P)

        # Contre l'aberration chromatique : patchs passés en gris au hasard
        gris = torch.rand(B, 9, 1, 1, 1, device=dev) < self.p_gris
        patchs = torch.where(gris, patchs.mean(dim=2, keepdim=True).expand_as(patchs), patchs)

        # Le centre (indice 4) avec chacun de ses 8 voisins ; la classe = le numéro du voisin
        idx_voisins = torch.tensor([3 * i + j for i, j in VOISINS], device=dev)
        centre = patchs[:, 4:5].expand(B, 8, C, P, P).reshape(-1, C, P, P)
        voisin = patchs[:, idx_voisins].reshape(-1, C, P, P)
        cibles = torch.arange(8, device=dev).repeat(B)
        return centre, voisin, cibles


def recalibrer_bn(model, loader, n_lots=100):
    """L'encodeur a vu des patchs 8 x 8 ; on recalcule ses moyennes BatchNorm sur des images entières."""
    for m in model.model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # moyenne cumulée sur tous les lots
    model.train()
    with torch.no_grad():
        for k, (images, _) in enumerate(loader):
            model.encode(images.cuda())
            if k + 1 >= n_lots:
                break
    model.eval()


def entrainer_position(epochs=30, lr=0.1, **reglages):
    """Entraîne la tâche prétexte sur CIFAR-10, SANS les étiquettes ; renvoie le modèle."""
    model = RelativePosition(**reglages).cuda()
    # Même prétraitement que la rotation (recadrage + miroir aléatoires)
    dset = datasets.CIFAR10('data/cifar10', train=True, download=True,
                            transform=get_transform('cifar10', 'rotation', train=True))
    loader = data.DataLoader(dset, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
    # Mêmes réglages que train_self_supervised_task.py de la librairie
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    for epoque in range(epochs):
        model.train()
        pertes, precisions = [], []
        for images, _ in loader:  # les étiquettes ne sont jamais utilisées
            metriques, _ = model(images.cuda())
            opt.zero_grad()
            metriques['Loss'].backward()
            opt.step()
            pertes.append(metriques['Loss'].item())
            precisions.append(metriques['Acc1'].item())
        sched.step()
        print(f'époque {epoque + 1:2d}/{epochs}  perte {sum(pertes) / len(pertes):.3f}  '
              f'tâche prétexte {sum(precisions) / len(precisions):5.1f} %')

    recalibrer_bn(model, loader)
    torch.save(model.state_dict(), 'position_relative.pth')
    return model


if __name__ == '__main__':
    entrainer_position(epochs=200)
