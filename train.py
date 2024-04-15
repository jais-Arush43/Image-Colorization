from tqdm.notebook import tqdm
from dataloader import *
from utils import *
def train_model(model, train_set, epochs, display_every=200):
    data = next(iter(val_set)) 
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() 
        i = 0                                
        for data in tqdm(train_set):
            model.setup_input(data)
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0))
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_set)}")
                log_results(loss_meter_dict) 
                visualize(model, data, save=False) 

