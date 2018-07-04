import nest
import numpy as np

# These neuron parameters lead to a iaf_psc_alpha neuron that fires with a
# constant rate of approximately f_out = I_e / 10.0
FIXED_RATE_NEURON = {'I_e': 0.,        # constant input
                     'tau_m': 82.,     # membrane time constant
                     'V_th': -55.,     # threshold potential
                     'E_L': -70.,      # membrane resting potential
                     't_ref': 2.,      # refractory period
                     'V_reset': -80.,  # reset potential
                     'C_m': 320.,      # membrane capacitance
                     'V_m': -70.}      # initial membrane potential

DLS2_NEURON = {'I_e': 0.,        # constant input
               'tau_m': 3.,      # membrane time constant
               'V_th': 1200.,    # threshold potential
               'E_L': 800.,      # membrane resting potential
               't_ref': 1.5,     # refractory period
               'V_reset': 600.,  # reset potential
               'C_m': 2.36,      # membrane capacitance
               'V_m': 800.}      # initial membrane potential

class Spikes():
    def __init__(self):
        self.nr_spikes_total = 0
        self.nr_spikes_prev = 0
        self.nr_spikes_new = 0
        self.spikes = {'senders': np.array([]),
                       'times': np.array([])}

class Unit(object):
    def __init__(self, unit_type = 'iaf_psc_alpha', dimension = 0, parameters = DLS2_NEURON):
        self.unit_type = unit_type
        self.dimension = dimension
        self.parameters = parameters
        self.node_ids = ()
        self.input_connections = []
        self.output_connections = []
        self.spike_detector = None
        self.spikes = Spikes()

    def get_spikes(self):
        self.spikes.nr_spikes_prev = self.spikes.nr_spikes_total
        self.spikes.spikes = nest.GetStatus(self.spike_detector, keys="events")[0]
        self.spikes.nr_spikes_total = len(self.spikes.spikes["senders"])
        self.spikes.nr_spikes_new = (self.spikes.nr_spikes_total - self.spikes.nr_spikes_prev)
        return self.spikes.spikes

    def get_new_spikes(self):
        self.get_spikes()
        nr_spikes_new = self.spikes.nr_spikes_new
        spikes = {"senders": self.spikes.spikes["senders"][-nr_spikes_new:],
                  "times": self.spikes.spikes["times"][-nr_spikes_new:]}
        return spikes

    def get_activity(self):
        senders = self.get_new_spikes()["senders"]
        activity = np.zeros(self.dimension)
        for idx, neuron_id in enumerate(self.node_ids):
            activity[idx] += len(np.where(np.array(senders)==neuron_id)[0])
        mean = np.mean(activity)
        if mean != 0.:
            activity /= mean*2
        return activity

class Model:
    def __init__(self):
        self.units = []
        self.input_unit = None

    def add(self, unit):
        unit.node_ids = nest.Create(unit.unit_type, unit.dimension)
        # create spike detector for the unit
        unit.spike_detector = nest.Create('spike_detector', unit.dimension,  params={"withgid": True, "withtime": True})
        nest.Connect(unit.node_ids, unit.spike_detector)
        self.units.append(unit)
    
    def build(self):
        units = self.units
        num_layers = len(units)

        for index, unit in enumerate(units):
            if (index < num_layers-1):
                initial_weights = 1000*np.random.randn(units[index+1].dimension, unit.dimension)
                nest.Connect(unit.node_ids, units[index+1].node_ids, 'all_to_all', {'weight': initial_weights})
                connection = nest.GetConnections(source=unit.node_ids, target=units[index+1].node_ids)
                unit.output_connections.append(connection)
                units[index+1].input_connections.append(connection)

        # create fixed rate input neurons
        self.input_unit = Unit(
            unit_type='iaf_psc_alpha', 
            dimension=units[0].dimension, 
            parameters=FIXED_RATE_NEURON
        )
        self.input_unit.node_ids = nest.Create(unit.unit_type, unit.dimension)
        # create spike detector for the unit
        self.input_unit.spike_detector = nest.Create(
            'spike_detector', 
            self.input_unit.dimension,  
            params={"withgid": True, "withtime": True}
        )
        nest.Connect(self.input_unit.node_ids, self.input_unit.spike_detector)

        # and connect them to the first layer
        # TODO(Christian): This should not be hardcoded here, but just another kind of layer.
        nest.Connect(self.input_unit.node_ids, units[0].node_ids, 'one_to_one', syn_spec={"weight":1000.0, "delay":1.0})
        connection = nest.GetConnections(source=self.input_unit.node_ids, target=units[0].node_ids)
        units[0].input_connections.append(connection)

    def get_spikes_at_layer(self, layer_index):
        return self.units[layer_index].get_spikes()

    def get_activity_at_input(self):
        return self.input_unit.get_activity()

    def get_spikes_at_input(self):
        return self.input_unit.get_spikes()

    def get_activity_at_layer(self, layer_index):
        return self.units[layer_index].get_activity()

    def get_weights_at_layer(self, layer_index):
        unit = self.units[layer_index]
        assert(len(unit.input_connections) == 1)
        assert(layer_index >= 1)

        # See http://www.nest-simulator.org/py_sample/plot_weight_matrices/index.html
        # There got to be a better way...
        connection = unit.input_connections[0]
        flattened_weights = nest.GetStatus(connection, keys='weight')

        input_dimension = self.units[layer_index-1].dimension
        output_dimension = unit.dimension
        weights = np.zeros(input_dimension, output_dimension)

        for idx, n in enumerate(connection):
            weights[n[0] - min(unit.node_ids), n[1] - min(unit.node_ids)] = flattened_weights[idx]

        return weights

    def set_weights_at_layer(self, layer_index, weight):
        """
        """
        unit = self.units[layer_index]

        assert(len(unit.input_connections) == 1)
        assert(layer_index >= 1)

        connection = unit.input_connections[0]
        nest.SetStatus(connection, {"weight": weight})

        ## This is an ULTRA HACK, there be a better
        ## way however (see above)
        #for i, source in enumerate(previous_unit.node_ids):
        #    for j, target in enumerate(unit.node_ids):
        #        synapse = nest.GetConnections(
        #            source=tuple((source,)),
        #            target=tuple((target,))
        #        )
        #        nest.SetStatus(synapse, {"weight": weight[i,j]})

    def set_stimulus(self, input):
        for i, n in enumerate(self.input_unit.node_ids):
            nest.SetStatus(tuple((n,)), {"I_e": 1000. * input[i]})

if __name__ == '__main__':
    m = Model()
    m.add(Unit(
        unit_type='iaf_psc_alpha',
        dimension=4
    ))
    m.add(Unit(
        unit_type='iaf_psc_alpha',
        dimension=20
    ))
    m.add(Unit(
        unit_type='iaf_psc_alpha',
        dimension=4
    ))
    m.build()

    m.set_stimulus([1000.0,2000.0,3000.0,3000.0])
    nest.Simulate(100)
    
    print(m.get_activity_at_input())
    print(m.get_activity_at_layer(0))
    print(m.get_activity_at_layer(1))
    print(m.get_activity_at_layer(2))


