
import os


def prepareRun(run_name):
    struct = {}
    struct['base'] = './models/' + run_name + '/'
    struct['run'] = run_name
    # Create the bast folder
    if not os.path.exists(struct['base']):
        os.makedirs(struct['base'])
    # Create the sub-directories
    subFolders = ['Tensorboard','Checkpoints','Saves']
    for s in subFolders:
        struct[s] = struct['base'] + s + '/'
        if not os.path.exists(struct[s]):
            os.makedirs(struct[s])
    return struct

