from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models

class StyleTransfer(object):
    def __init__(self):

        # initialize images as None to flags
        self.content_img = None
        self.style_img = None
        self.target_img = None

        # initialize model as None to flag
        self.model = None

        # check for gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def load_content(self, image_path, max_size=400, shape=None):
        """ Load content image """

        self.content_img = self._load_image(image_path, max_size=max_size, shape=shape).to(self.device)


    def load_style(self, image_path, max_size=400, shape=None):
        """ Load style image """

        # check if content has been loaded
        self._flag_var(content=True)
        
        shape = self.content_img.shape[-2:]
        self.style_img = self._load_image(image_path, max_size=max_size, shape=shape).to(self.device)


    def load_model(self, print_model=False):
        """ Load vgg19 model. """

        self.model = models.vgg19(pretrained=True).features
        self.model.to(self.device)

        # freeze all VGG parameters since we're only optimizing the target image
        for param in self.model.parameters():
            param.requires_grad_(False)

        if print_model:
            print(self.model)


    def transfer(self, style_weights, learning_rate=5e-3, epochs=2000, content_weight=1, style_weight=1e6):
        """ Transfer style from style_img to content_img. """

        # check if model, content_img and style_img have been loaded
        self._flag_var(model=True, content=True, style=True)

        # get content and style features only once before forming the target image
        content_features = self._get_features(self.content_img, self.model)
        style_features = self._get_features(self.style_img, self.model)

        # calculate the gram matrices for each layer of our style representation
        style_grams = {layer: self._gram_matrix(style_features[layer]) for layer in style_features}

        # create a third "target" image and prep it for change
        self.target_img = self.content_img.clone().requires_grad_(True).to(self.device)

        # for displaying the target image, intermittently
        show_every = 400

        # Adam optimizer
        optimizer = optim.Adam([self.target_img], lr=learning_rate)

        for ii in range(1, epochs+1):

            ## Content loss
            # get the features from your target image    
            target_features = self._get_features(self.target_img, self.model)
            # calculate the content loss
            content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

            ## Style loss
            style_loss = 0
            # iterate through each style layer and add to the style loss
            for layer in style_weights:
                # get the "target" style representation for the layer
                target_feature = target_features[layer]
                _, d, h, w = target_feature.shape

                # Calculate the target gram matrix
                target_gram = self._gram_matrix(target_feature)

                # get the "style" style representation
                style_gram = style_grams[layer]

                # Calculate the style loss for one layer, weighted appropriately
                layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)

                # add to the style loss
                style_loss += layer_style_loss / (d * h * w)

            ## Total loss
            total_loss = content_weight * content_loss + style_weight * style_loss

            # update your target image
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # display intermediate images and print the loss
            if  ii % show_every == 0:
                print('Total loss: ', total_loss.item())
                plt.imshow(self._im_convert(self.target_img))
                plt.show()


    def show_images(self, content=True, style=True, target=False, figsize=(20,10)):
        """ Show images """

        self._flag_var(content=content, style=style, target=target)

        imgs_list = [self.content_img, self.style_img, self.target_img]
        bool_list = [content, style, target]

        plot_list = [imgs_list[i] for i in range(3) if bool_list[i]]

        fig = plt.figure(figsize=figsize)

        for i in range(len(plot_list)):
            plt.subplot(1,len(plot_list),i+1)
            plt.imshow(self._im_convert(plot_list[i]))


    def _get_features(self, image, model, layers=None):
        """ Run an image forward through a model and get the features for 
            a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
        """

        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2', # content representation
                  '28': 'conv5_1'}
                        
        features = {}
        x = image

        # model._modules is a dictionary holding each module in the model
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
                
        return features


    def _gram_matrix(self, tensor):
        """ Calculate the Gram Matrix of a given tensor 
            Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
        """
        
        ## get the batch_size, depth, height, and width of the Tensor
        _, depth, height, width = tensor.size()
        ## reshape it, so we're multiplying the features for each channel
        tensor = tensor.view(depth, height * width)
        ## calculate the gram matrix
        gram = torch.mm(tensor, tensor.t())
        
        return gram 


    def _load_image(self, img_path, max_size=400, shape=None):
        """ Load in and transform an image, making sure the image
        is <= 400 pixels in the x-y dims. """
        
        image = Image.open(img_path).convert('RGB')
        
        # large images will slow down processing
        if max(image.size) > max_size:
            size = max_size
        else:
            size = max(image.size)
        
        if shape is not None:
            size = shape
            
        in_transform = transforms.Compose([
                            transforms.Resize(size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), 
                                                (0.229, 0.224, 0.225))])

        # discard the transparent, alpha channel (that's the :3) and add the batch dimension
        image = in_transform(image)[:3,:,:].unsqueeze(0)
        
        return image


    def _im_convert(self, tensor):
        """ Display a tensor as an image. """
        
        image = tensor.to("cpu").clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1,2,0)
        image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        image = image.clip(0, 1)

        return image

    
    def _flag_var(self, model=False, content=False, style=False, target=False):
        """ Verifies that the variable has been loaded. """

        if model and self.model is None:
            print("VGG19 not loaded!")
            return -1

        if content and self.content_img is None:
            print("Content image not loaded!")
            return -1

        if style and self.style_img is None:
            print("Style image not loaded!")
            return -1

        if target and self.target_img is None:
            print("Target image not found!")
            return -1