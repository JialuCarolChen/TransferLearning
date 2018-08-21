### The project includes the following files:
- clr_callback.py: a Keras callback implementation to run the cyclical learning rate policies:
https://github.com/bckenstler/CLR 
- imageLoader.py: a module to load data for modelling (work with train.py)
- DataConvert.py: a module built to convert the data directory to the structure that fits with the Keras flow_from_directory function
- TransLearnModel.py: a module built to train a transfer learning model 
- train.py: a script uses TransLearnModel to solve a image classificaton task with transfer learning 
- train_flow.py: a script uses TransLearnModel to solve a image classificaton task with or without transfer learning, data flow from directory  
- image augmentation.ipynb: a jupyter notebook to show examples of image augmentation
- predict and analyse.ipynb: a jupyter notebook to produce the prediction file and analyse the prediction results 

### Instructions on running the code:
##### 1. Go to file train_flow.py and specify the following parameters:
- dir_path: (string) the path of the data folder 
- file_prefix: (string) the prefix of the name of the output files
- model_type: (string) the type of the model architecture, available options are 'vgg16', 'vgg19', 'resnet50' and 'xception'
- use_es: (bool) whether to use early stopping or not
- epochs: (int) maximum number of training epochs to run
- use_clr: (bool) whether to use cyclical learning rate
- use_decay_lr: (bool) whether to use drop decay learning rate
- lr: (float) learning rate, if cyclical learning rate is used, it's the minimum learning rate; if step decay learning rate is used, it's the initial learning rate
- bsize: (int) batch size
- is_tl: (bool) whether to apply transfer learning or not
##### 2. Run the script train_flow.py (To train without flow from directory, use train.py)
$ python train_flow.py

##### 3. To produce the prediction file and analyse the prediction results: 
Open jupyter notebook and run the code in prediction_analysis.ipynb.  


 
