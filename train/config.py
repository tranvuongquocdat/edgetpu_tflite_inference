class Config:
    def __init__(self):
        # Dataset configs
        self.dataset_type = "slr"  # 'slr' or 'open_images'
        self.datasets = ["vision/datasets/gud_dataset_merge"]  # Đường dẫn đến thư mục chứa dữ liệu train
        self.validation_dataset = "vision/datasets/gud_dataset_merge"  # Đường dẫn đến thư mục chứa dữ liệu validation
        self.balance_data = False

        # Network configs
        self.net = "mb3-ssd-lite"  # 'mb2-ssd-lite' or 'mb3-ssd-lite'
        self.freeze_base_net = False
        self.freeze_net = False
        self.mb2_width_mult = 1.0

        # Training params
        self.lr = 1e-3
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.gamma = 0.1
        self.base_net_lr = None
        self.extra_layers_lr = None

        # Model loading
        self.base_net = None
        self.pretrained_ssd = None
        self.resume = None

        # Scheduler
        self.scheduler = "multi-step"  # 'multi-step' or 'cosine'
        self.milestones = "80,100"
        self.t_max = 120

        # Training configs
        self.batch_size = 32
        self.num_epochs = 120
        self.num_workers = 0
        self.validation_epochs = 5
        self.debug_steps = 100
        self.use_cuda = True

        # Paths
        self.checkpoint_folder = 'trained_models/'

# Default config
default_config = Config() 