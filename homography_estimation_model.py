import torch
import torch.nn as nn
import torch.nn.functional as F

class HomographyModel(nn.Module):
    """
    Deep Homography Estimation Model.
    Attributes
    ----------
        * Layers :
            - Convolutional layers -> layer1-layer8
            - Dropout -> drop1
            - Fully connected layer -> fc1
            - Dropout -> drop2
            - Fully connected layer -> fc2
    Methods
    -------
        * Forward(): Forward pass.
    """

    def __init__(self):
        """
        Constructor.
        """
        
        super().__init__()

        # ------------ Convolutional layers -----------
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding = 'same') ,
            nn.ReLU()                             ,
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding = 'same') ,
            nn.ReLU()                              ,
            nn.BatchNorm2d(64)                     ,
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding = 'same') ,
            nn.ReLU()                              ,
            nn.BatchNorm2d(64)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding = 'same') ,
            nn.ReLU()                              ,
            nn.BatchNorm2d(64)                     ,
            nn. MAxPool2d(2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding = 'same') ,
            nn.ReLU()                               ,
            nn.BatchNorm2d(128)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding = 'same') ,
            nn.ReLU()                                ,
            nn.BatchNorm2d(128)                      ,
            nn. MAxPool2d(2)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding = 'same') ,
            nn.ReLU()                                ,
            nn.BatchNorm2d(128)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding = 'same') ,
            nn.ReLU()                                ,
            nn.BatchNorm2d(128)
        )
        # ---------------------------------------------
        
        # ------ Fully connected + drop out -----------
        self.drop1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128*16*16, 1024)
        self.drop2 = nn.Dropout(0.5)
        self. fc2 = nn.Linear(1024, 8)
         # ---------------------------------------------
    
    def forward(self, x):
        """
        Forward pass.
        Input
        -----
            * x : input (2 channel image).
        """

        # --- Convolutional layers ---
        out = self.layer1(x)
        out = self.layer2(x)
        out = self.layer3(x)
        out = self.layer4(x)
        out = self.layer5(x)
        out = self.layer6(x)
        out = self.layer7(x)
        out = self.layer8(x)
        # ----------------------------
 
        # Flatten
        out = out.view(-1, 128*16*16, 1024)

        # --- Fully connected + drop out ---
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.fc2(out)
        # ----------------------------------
        
        return out
        