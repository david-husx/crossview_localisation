# CVM-Net: Cross-View Matching Network for Image-Based Ground-to-Aerial Geo-Localisation

TO BE COMPLETED...


This project is an initial research on image-based geo-localisation with satellite imagery as reference map. The problem is regarded as image retrieval problem: to query a ground-level image from geo-tagged satelite image database. We propose a deep learning framework to extract global descriptors of ground and satellite images. VGG16 is used to extract local features and NetVLAD layers is used to aggregate to global descriptors. A Siamese-like architecture is used to train the network. To query a grounod-level image, we retrieve the satellite image with smallest distance between desscriptor of the query image and the satellite image. The position (latitude and longitude) of satellite image center is the position of the query.

### Abstract
The problem of localization on a geo-referenced aerial/satellite map given a query ground view image remains challenging due to the drastic change in viewpoint that causes traditional image descriptors based matching to fail. We leverage on the recent success of deep learning to propose the CVM-Net for the cross-view image-based ground-to-aerial geo-localization task. Specifically, our network is based on the Siamese architecture to do metric learning for the matching task. We first use the fully convolutional layers to extract local image features, which are then encoded into global image descriptors using the powerful NetVLAD. As part of the training procedure, we also introduce a simple yet effective weighted soft margin ranking loss function that not only speeds up the training convergence but also improves the final matching accuracy. Experimental results show that our proposed network significantly outperforms the state-of-the-art approaches on two existing benchmarking datasets.

### Network Architecture


### Experiment Dataset


### Results


### Models


### Publications



### Acknowledgement
This work is finished in Department of Computer Science, National University of Singapore.
