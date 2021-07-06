import json


class TrainParameters:
    """ Store the parameters that used in training """

    """ Device ID """
    DEV_IDS = [2]

    """ Verbose mode (Debug) """
    VERBOSE_MODE = False

    """ Max epochs when training """
    MAX_EPOCHS = 4

    """ Mini-Batch Size """
    LOADER_BATCH_SIZE = 6               # for training batch size

    LOADER_VALID_BATCH_SIZE = 1    # for validation batch size

    """ Pytorch Dataloader  """
    LOADER_NUM_THREADS = 4              # threads

    LOADER_SHUFFLE = True               # data loader shuffle enabled

    LOADER_PIN_MEM = True               # should not be changed this parameter

    """ Learning Rate """
    START_LR = 1e-4

    """ Learning rate decay """
    LR_DECAY_FACTOR = 0.5

    """ Learning rate decay steps """
    LR_DECAY_STEPS = 1                  # epochs

    """ Logging Steps """
    LOG_STEPS = 5               # per training-iterations

    """ Validation Steps """
    VALID_STEPS = 200           # per training-iterations

    """ Visualization Steps """
    VIS_STEPS = 100             # per training-iterations

    """ Validation Maximum Batch Number """
    MAX_VALID_BATCHES_NUM = 50

    """ Checkpoint Steps (iteration) """
    CHECKPOINT_STEPS = 5000      # per training-iterations

    """ Continue Step (iteration) """
    LOG_CONTINUE_STEP = 0            # continue step, used for logger

    """ Continue from (Dir) """
    LOG_CONTINUE_DIR = ''           # continue dir

    TQDM_PROGRESS = True

    """ Description """
    NAME_TAG = ''                   # short tag to describe training process

    DESCRIPTION = ''                # train description, used for logging changes

    def __init__(self, from_json_file=None):
        if from_json_file is not None:
            with open(from_json_file) as json_data:
                params = json.loads(json_data.read())
                json_data.close()

                # Extract parameters
                self.DEV_IDS = params['dev_id']
                self.MAX_EPOCHS = int(params['max_epochs'])
                self.LOADER_BATCH_SIZE = int(params['loader_batch_size'])
                self.LOADER_VALID_BATCH_SIZE = int(params['loader_valid_batch_size'])
                self.LOADER_NUM_THREADS = int(params['loader_num_threads'])
                self.LOADER_SHUFFLE = bool(params['loader_shuffle'])
                self.START_LR = float(params['start_learning_rate'])
                self.LR_DECAY_FACTOR = float(params['lr_decay_factor'])
                self.LR_DECAY_STEPS = int(params['lr_decay_epoch_size'])
                self.VERBOSE_MODE = bool(params['verbose'])
                self.VALID_STEPS = int(params["valid_per_batches"])
                self.MAX_VALID_BATCHES_NUM = int(params["valid_max_batch_num"])
                self.CHECKPOINT_STEPS = int(params['checkpoint_per_iterations'])
                self.VIS_STEPS = int(params['visualize_per_iterations'])
                self.LOG_CONTINUE_DIR = str(params['log_continue_dir'])
                self.LOG_CONTINUE_STEP = int(params['log_continue_step'])
                self.DESCRIPTION = str(params['description'])
                self.NAME_TAG = str(params['name_tag'])

    def extract_dict(self):
        params = dict()
        params['dev_id'] = self.DEV_IDS
        params['max_epochs'] = self.MAX_EPOCHS
        params['loader_batch_size'] = self.LOADER_BATCH_SIZE
        params['loader_valid_batch_size'] = self.LOADER_VALID_BATCH_SIZE
        params['loader_shuffle'] = self.LOADER_SHUFFLE
        params['start_learning_rate'] = self.START_LR
        params['lr_decay_factor'] = self.LR_DECAY_FACTOR
        params['lr_decay_epoch_size'] = self.LR_DECAY_STEPS
        params['loader_num_threads'] = self.LOADER_NUM_THREADS
        params['verbose'] = self.VERBOSE_MODE
        params['valid_per_batches'] = self.VALID_STEPS
        params['valid_max_batch_num'] = self.MAX_VALID_BATCHES_NUM
        params['checkpoint_per_iterations'] = self.CHECKPOINT_STEPS
        params['visualize_per_iterations'] = self.VIS_STEPS
        params['log_continue_step'] = self.LOG_CONTINUE_STEP
        params['description'] = self.DESCRIPTION
        params['name_tag'] = self.NAME_TAG
        params['log_continue_dir'] = self.LOG_CONTINUE_DIR
        return params

    def save(self, json_file_path):
        params = self.extract_dict()
        with open(json_file_path, 'w') as out_json_file:
            json.dump(params, out_json_file, indent=2)

    def report(self):
        params = self.extract_dict()
        for param_key in params.keys():
            print(param_key, ': ', str(params[param_key]))

    def to_json(self):
        params = self.extract_dict()
        return json.dumps(params, indent=2)