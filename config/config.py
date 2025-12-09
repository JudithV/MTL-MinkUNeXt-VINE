import yaml
import os
current_directory = os.path.dirname(os.path.realpath(__file__))
class Config():
    def __init__(self, yaml_file=os.path.join(current_directory, 'general_parameters.yaml')):
        # print(current_directory)
        # print(os.path.join(current_directory, 'general_parameters.yaml'))
        # # print(os.path.dirname(os.path.realpath(__file__)))
        # print(os.getcwd())
        with open(yaml_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            print(config)

            self.dataset_folder= config.get('dataset_folder')
            self.cuda_device = config.get('cuda_device')

            self.patience = config.get('patience')
            self.min_delta = config.get('min_delta')

            self.quantization_size = config.get('quantization_size')
            self.num_workers = config.get('num_workers')
            self.batch_size = config.get('batch_size')
            self.batch_size_limit = config.get('batch_size_limit')
            self.batch_expansion_rate = config.get('batch_expansion_rate')
            self.batch_expansion_th = config.get('batch_expansion_th')
            self.batch_split_size = config.get('batch_split_size')
            self.val_batch_size = config.get('val_batch_size')

            self.format_point_cloud = config.get('format_point_cloud')
            
            self.spherical_coords = config.get('spherical_coords')
            self.normalize = config.get('normalize')
            self.use_intensity = config.get('use_intensity')
            self.equalize_intensity = config.get('equalize_intensity')
            self.process_intensity = config.get('process_intensity')
            self.correct_intensity = config.get('correct_intensity')
            self.use_downsampled = config.get('use_downsampled')
            self.min_distance = config.get('min_distance')
            self.max_distance = config.get('max_distance')

            self.use_2D = config.get('use_2D')
            
            self.optimizer = config.get('optimizer')
            self.initial_lr = config.get('initial_lr')
            self.scheduler = config.get('scheduler')
            self.aug_mode = config.get('aug_mode')
            self.weight_decay = config.get('weight_decay')
            self.loss = config.get('loss')
            self.margin = config.get('margin')
            self.tau1 = config.get('tau1')
            self.positives_per_query = config.get('positives_per_query')
            self.similarity = config.get('similarity')
            self.pos_margin = config.get('pos_margin')
            self.neg_margin = config.get('neg_margin')
            self.normalize_embeddings = config.get('normalize_embeddings')

            self.aggregator_fusion = config.get('aggregator_fusion')
            self.clustering_head = config.get('clustering_head')
            self.clustering_importance = float(config.get('clustering_importance'))
            self.cluster_batch_size = config.get('cluster_batch_size')
            self.use_cross_entropy = config.get('use_cross_entropy')
            self.cross_entropy_importance = config.get('cross_entropy_importance')
            self.labeling_head = config.get('labeling_head')
            self.labeling_importance = float(config.get('labeling_importance'))
            self.labeling_batch_size = config.get('labeling_batch_size')

            self.protocol = config.get('protocol')

            if self.protocol == 'baseline':
                self.epochs = config.get('baseline').get('epochs')
                self.scheduler_milestones = config.get('baseline').get('scheduler_milestones')
                self.num_points = config.get('baseline').get('num_points')
                self.train_file = config.get('baseline').get('train_file')
                self.val_file = config.get('baseline').get('val_file')
            elif self.protocol == 'refined':
                self.epochs = config.get('refined').get('epochs')
                self.scheduler_milestones = config.get('refined').get('scheduler_milestones')
                self.num_points = config.get('refined').get('num_points')
                self.train_file = config.get('refined').get('train_file')
                self.val_file = config.get('refined').get('val_file')
            elif self.protocol == 'arvc':
                self.epochs = config.get('arvc').get('epochs')
                self.scheduler_milestones = config.get('arvc').get('scheduler_milestones')
                self.num_points = config.get('arvc').get('num_points')
                self.train_file = config.get('arvc').get('train_file')
                self.val_file = config.get('arvc').get('val_file')
            elif self.protocol == 'intensityOxford':
                self.epochs = config.get('intensityOxford').get('epochs')
                self.scheduler_milestones = config.get('intensityOxford').get('scheduler_milestones')
                self.num_points = config.get('intensityOxford').get('num_points')
                self.train_file = config.get('intensityOxford').get('train_file')
                self.val_file = config.get('intensityOxford').get('val_file')        
            elif self.protocol == "usyd":
                self.epochs = config.get('usyd').get('epochs')
                self.scheduler_milestones = config.get('usyd').get('scheduler_milestones')
                self.num_points = config.get('usyd').get('num_points')
                self.train_file = config.get('usyd').get('train_file')
                self.val_file = config.get('usyd').get('val_file')
            elif self.protocol == "blt":
                self.epochs = config.get('blt').get('epochs')
                self.scheduler_milestones = config.get('blt').get('scheduler_milestones')
                self.num_points = config.get('blt').get('num_points')
                self.train_file = config.get('blt').get('train_file')
                self.val_file = config.get('blt').get('val_file')
            elif self.protocol == "vmd":
                self.epochs = config.get('vmd').get('epochs')
                self.scheduler_milestones = config.get('vmd').get('scheduler_milestones')
                self.num_points = config.get('vmd').get('num_points')
                self.train_file = config.get('vmd').get('train_file')
                self.val_file = config.get('vmd').get('val_file')

            self.print_model_info = config.get('print').get('model_info')
            self.print_model_parameters = config.get('print').get('number_of_parameters')
            self.debug = config.get('print').get('debug')
            self.weights_path = config.get('evaluate').get('weights_path')

PARAMS = Config()


