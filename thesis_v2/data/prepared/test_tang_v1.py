import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from thesis_v2.data.prepared import tang_v1
from thesis_v2 import dir_dict

global_dict = {
    'img_path': join(dir_dict['private_data'],
                     'tang_v1',
                     'images',
                     ),
    'num_img_train': 34000,
    'num_img_val': 1000,
    'num_neurons_total': 302,
    # neurons with responses to
    'num_neurons_with_response_to_all_images': 302,
#     'num_trial': 4,
}

x_train = np.load(join(global_dict['img_path'], 'trainPic_m1s1.npy'))

train_img100 = tang_v1.images(x_train, px_kept=100, final_size=50)
train_img80 = tang_v1.images(x_train)

plt.imshow(train_img100[0][0],cmap='gray')
plt.savefig('train_img100.png')
plt.show(train_img80[0][0], cmap='gray')
plt.savefig('train_img80.png')