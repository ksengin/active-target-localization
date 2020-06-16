import torch

class InformationAccumulator():
    def __init__(self, init_sensor_loc:torch.tensor=None, target_loc:torch.tensor=None):
        if init_sensor_loc is not None and target_loc is not None:
            self.reset(init_sensor_loc, target_loc)        

    # reset all information
    def reset(self, init_sensor_loc: torch.tensor, target_loc: torch.tensor):
        self.target_pos = target_loc.reshape(-1,2) 
        self.total_information_determinant = 0.
        
        self.measurement_locs = init_sensor_loc
        self.sensor_to_target_vectors = self.target_pos.unsqueeze(1) - self.measurement_locs
        self.angles = torch.atan2(self.sensor_to_target_vectors[:,:,1], self.sensor_to_target_vectors[:,:,0])
        self.distances = torch.norm(self.sensor_to_target_vectors, p=2, dim=-1)

    # compute the fisher information determinant
    def fisher_determinant(self, next_meas_loc: torch.tensor):
        loc_to_target = self.target_pos - next_meas_loc
        angles = torch.atan2(loc_to_target[:,1], loc_to_target[:,0])
        dists = torch.norm(loc_to_target, p=2, dim=1)

        fisher_contrib = torch.sin(self.angles - angles)**2 / (self.distances**2 * dists**2)
        fisher_contrib = fisher_contrib.sum()
        
        self.total_information_determinant += fisher_contrib
        self.add_measurement_location(next_meas_loc)

        return self.total_information_determinant

    # add the next measurement location to the list of sensor locations
    def add_measurement_location(self, next_meas_loc: torch.tensor):
        self.measurement_locs = torch.cat((self.measurement_locs.reshape(-1,2), next_meas_loc.reshape(-1,2)))

        loc_to_target = self.target_pos - next_meas_loc
        angle = torch.atan2(loc_to_target[:,1], loc_to_target[:,0])
        dist = torch.norm(loc_to_target, p=2, dim=1)
        self.angles = torch.cat((self.angles, angle.reshape(-1,1)), 1)
        self.distances = torch.cat((self.distances, dist.reshape(-1,1)), 1)
