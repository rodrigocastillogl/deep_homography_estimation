import torch
import torch.nn as nn
import torch.nn.functional as F

class HomographyModel(nn.Module):
    """
    Deep Homography Estimation Model.
    Attributes
    ----------
    Methods
    -------
    """

    def __init__(self):
        """
        Constructor.
        Input
        -----
        Output
        ------
        """
        super().__init__()

        self.layer1 = nn.Sequential( nn.Conv2d(2, 64, 3, padding = 'same') ,
                      nn.ReLU()                                            ,
                      nn.BatchNorm2d(64)                                   )

        self.layer2 = nn.Sequential( nn.Conv2d(64, 64, 3, padding = 'same') ,
                      nn.ReLU()                                             ,
                      nn.BatchNorm2d(64)                                    ,
                      nn.MaxPool2d(2)                                       )

        self.layer3 = nn.Sequential( nn.Conv2d(64, 64, 3, padding = 'same') ,
                      nn.ReLU()                                             ,
                      nn.BatchNorm2d(64)                                    )

        self.layer4 = nn.Sequential( nn.Conv2d(64, 64, 3, padding = 'same') ,
                      nn.ReLU()                                             ,
                      nn.BatchNorm2d(64)                                    ,
                      nn. MAxPool2d(2)                                      )
        
        self.layer5 = nn.Sequential( nn.Conv2d(64, 128, 3, padding = 'same') ,
                      nn.ReLU()                                              ,
                      nn.BatchNorm2d(128)                                    )
        
        self.layer6 = nn.Sequential( nn.Conv2d(128, 128, 3, padding = 'same') ,
                      nn.ReLU()                                               ,
                      nn.BatchNorm2d(128)                                     ,
                      nn. MAxPool2d(2)                                        )
        
        self.layer7 = nn.Sequential( nn.Conv2d(128, 128, 3, padding = 'same') ,
                      nn.ReLU()                                               ,
                      nn.BatchNorm2d(128)                                     )
        
        self.layer8 = nn.Sequential( nn.Conv2d(128, 128, 3, padding = 'same') ,
                      nn.Linear()                                               ,
                      nn.BatchNorm2d(128)                                    )