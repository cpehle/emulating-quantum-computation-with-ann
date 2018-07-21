import numpy as np
import pandas as pd

from pyhalbe import HICANN
from pyhalbe.HICANN import SynapseWeight
from pymarocco import PyMarocco, Defects
from pymarocco import runtime
from pymarocco.coordinates import LogicalNeuron
from pymarocco.results import Marocco

import pyhalbe
import pyhalbe.Coordinate as C
import pyhmf as pynn
import pysthal

import copy

class UpdateAnalogOutputConfigurator(pysthal.HICANNv4Configurator):
    """ Configures the following things from sthal container:
        - neuron quad configuration
        - analog readout
        - analog current input
        - current stimulus strength/duration etc.
    """
    def config_fpga(self, *args):
        """do not reset FPGA"""
        pass

    def config(self, fpga_handle, h, hicann):
        """Call analog output related configuration functions."""
        self.config_neuron_config(h, hicann)
        self.config_neuron_quads(h, hicann)
        self.config_analog_readout(h, hicann)
        self.config_fg_stimulus(h, hicann)
        self.flush_hicann(h)

# magic number from marocco
SYNAPSE_DECODER_DISABLED_SYNAPSE = HICANN.SynapseDecoder(1)

def update_synapse_on_wafer(wafer, synapse, weight, decoder):
    """Update a synapse to a specific weight and decoder.

    Args:
        wafer: Wafer on which the synapse is placed.
        synapse: Synapse of which the weight is to be set.
        weight: Weight to be set (of type SynapseWeight).
        decoder: Decoder to be used.
    """
    # TODO(Christian): How expensive is synapse.toHICANNOnWafer()
    proxy = wafer[synapse.toHICANNOnWafer()].synapses[synapse]
    decoder = copy.copy(decoder) # avoid overriding the decoder I guess?
    if value == 0:
        decoder = SYNAPSE_DECODER_DISABLED_SYNAPSE
    weight = SynapseWeight(value)
    proxy.decoder = decoder
    proxy.weight = weight

def get_synapse_on_wafer(wafer, synapse):
    """Read a synaptic weight from the wafer.

    Args:
        wafer: Wafer to be used.
        synapse: Synapse to be read.
    """
    proxy = wafer[synapse.toHICANNOnWafer()].synapses[synapse]
    return proxy.weight, proxy.decoder



def extract_synapse_placement_of_projection(placement_results, wafer, projection):
    """Return for each projection in the bio graph a set of hardware 
    synapses associated with it.

    Args:
        placement_results: Results produced by the place and route algorithm.
        wafer: Wafer on which the hardware synapse was placed.
        projection: Projection in the bio graph.
    """
    proj_items = placement_results.synapse_routing.synapses().find(projection)
    for synapse_on_wafer in proj_items:
        print(item)
    #if len(proj_items) == 0:
    #    return None, None
    #proj_item, = proj_items
    #self.synapses[proj] = synapse = proj_item.hardware_synapse()
    #decoder = copy.copy(wafer[synapse.toHICANNOnWafer()].synapses[synapse].decoder)


def configure_marocco_parameters(marocco, wafer, calibration_path):
    """Configure marroco to have the correct parameters.

    Note: This is not a generic way of configuring marocco. It
          just happens to be what was used in the model-deep-loop
          repository.

    Args:
        marocco: Instance of PyMarocco to use.
        wafer (C.Wafer): Wafer coordinate to run on.
        calibration_path (str): Path to the calibration database to use.
    """
    marocco.neuron_placement.default_neuron_size(4)
    marocco.neuron_placement.minimize_number_of_sending_repeaters(False)
    marocco.merger_routing.strategy(marocco.merger_routing.one_to_one)
    marocco.l1_address_assignment.strategy(marocco.l1_address_assignment.low_first)
    try:
        marocco.experiment.truncate_membrane_traces(False)
        marocco.experiment.truncate_spike_times(False)
    except AttributeError:
        pass

    marocco.bkg_gen_isi = 125
    marocco.pll_freq = 125e6
    marocco.backend = PyMarocco.Hardware
    marocco.verification = PyMarocco.Skip
    marocco.calib_backend = PyMarocco.XML
    marocco.calib_path = calibration_path
    marocco.defects.path = calibration_path
    marocco.defects.backend = Defects.XML
    marocco.default_wafer = wafer
    marocco.param_trafo.use_big_capacitors = True
    marocco.input_placement.consider_firing_rate(True)
    marocco.input_placement.bandwidth_utilization(0.8)


def configure_sthal_parameters(wafer, gmax, gmax_div, I_pl=None, min_V_syntc=None):
    """Configure certain analog parameters on all allocated HICANNs.

    Args:
        wafer: Wafer on which the HICANNs are located.
        gmax: Some Conductance probably.
        gmax_div: Who knows?
        I_pl: Who knows?
        min_V_syntc: Who knows?
    """
    for hicann in wafer.getAllocatedHicannCoordinates():
        fgs = wafer[hicann].floating_gates
        for ii in xrange(fgs.getNoProgrammingPasses()):
            cfg = fgs.getFGConfig(C.Enum(ii))
            cfg.fg_biasn = 0
            cfg.fg_bias = 0
            fgs.setFGConfig(C.Enum(ii), cfg)

        if I_pl is not None:
            for denmem in C.iter_all(C.NeuronOnHICANN):
                fgs.setNeuron(denmem, pyhalbe.HICANN.neuron_parameter.I_pl, I_pl)

        if min_V_syntc is not None:
            for param in [pyhalbe.HICANN.neuron_parameter.V_syntcx,
                          pyhalbe.HICANN.neuron_parameter.V_syntci]:
                for denmem in C.iter_all(C.NeuronOnHICANN):
                    val = fgs.getNeuron(denmem, param)
                    if val < min_V_syntc:
                        fgs.setNeuron(denmem, param, min_V_syntc)

        for block in C.iter_all(C.FGBlockOnHICANN):
            fgs.setShared(block, pyhalbe.HICANN.shared_parameter.V_gmax0, gmax)
            fgs.setShared(block, pyhalbe.HICANN.shared_parameter.V_gmax1, gmax)
            fgs.setShared(block, pyhalbe.HICANN.shared_parameter.V_gmax2, gmax)
            fgs.setShared(block, pyhalbe.HICANN.shared_parameter.V_gmax3, gmax)

        for block in C.iter_all(C.FGBlockOnHICANN):
            fgs.setShared(block, pyhalbe.HICANN.shared_parameter.V_dllres, 275)
            fgs.setShared(block, pyhalbe.HICANN.shared_parameter.V_ccas, 800)

        for driver in C.iter_all(C.SynapseDriverOnHICANN):
            for row in C.iter_all(C.RowOnSynapseDriver):
                wafer[hicann].synapses[driver][row].set_gmax_div(
                    C.left, gmax_div)
                wafer[hicann].synapses[driver][row].set_gmax_div(
                    C.right, gmax_div)


def initial_config_run(
    ctx, 
    marocco, 
    wafer, 
    gmax, 
    gmax_div, 
    I_pl, 
    min_V_syntc,
    configurator=None
    ):
    """Create an initial configuration of the Wafer. Runs place and 
    route and sets analog parameters.  

    Args:
        ctx: pynn context this is executed in (it is a singleton but nevermind).
        marocco: Instance of PyMarocco to be used.
        wafer: Wafer the place and route algorithm is to be executed on.
        gmax: ...
        gmax_div: ... (these should probably be HICANN / Synapse driver specific)
        I_pl: ...
        min_V_syntc: ...
        configurator: ...
    """
    marocco.skip_mapping = False
    marocco.backend = PyMarocco.None

    ctx.reset()
    ctx.run()

    configure_sthal_parameters(
        wafer=wafer,
        gmax=gmax,
        gmax_div=gmax_div,
        I_pl=I_pl,
        min_V_syntc=min_V_syntc
    )

    marocco.skip_mapping = True
    marocco.backend = PyMarocco.Hardware
    marocco.hicann_configurator = configurator or PyMarocco.HICANNv4Configurator

    ctx.reset()
    ctx.run()

    marocco.hicann_configurator = getattr(PyMarocco, "ParallelHICANNNoResetNoFGConfigurator",PyMarocco.NoResetNoFGConfigurator)


def measure_resting_potential(wafer, runtime_results, duration_in_ms, **kwargs):
    wafer.connect(pysthal.MagicHardwareDatabase())
    indices = []
    by_hicann = {}
    placement = runtime_results.placement
    for layer, layer_neurons in enumerate(self.neurons):
        for ii, pop in enumerate(layer_neurons):
            pl_item, = placement.find(pop[0])
            nrn = pl_item.logical_neuron().front()
            hicann = nrn.toHICANNOnWafer()
            by_hicann.setdefault(hicann, []).append((layer, ii, pop, nrn))
            indices.append((layer, ii))

    index = pd.MultiIndex.from_tuples(indices, names=("layer", "neuron"))
    data = pd.DataFrame(dict(v_rest=np.zeros(len(indices))), index=index)

    for hicann, items in by_hicann.iteritems():
        self.log("info", "prerun on {}", hicann)
        hicann_cfg = wafer_cfg[hicann]
        aout = C.AnalogOnHICANN(0)
        adc = hicann_cfg.analogRecorder(aout)
        adc.setRecordingTime(duration_in_ms * 1e-7)

        for layer, ii, pop, nrn in items:
            hicann_cfg.enable_aout(nrn, aout)
            wafer_cfg.configure(UpdateAnalogOutputConfigurator())
            adc.record()
            trace = adc.trace()
            voltage = (trace - 1.2) * 1e3 / 10.
            time = (adc.getTimestamps() - 20e-6) * 1e7
            membrane = np.vstack((time, voltage)).T
            self.log(
                "debug", "mean trace: {} {} -> {} ({} samples)",
                hicann, nrn, np.mean(trace), len(trace))
            hicann_cfg.disable_aout()

            v_rest = np.mean(voltage)
            v_rest_std = np.std(voltage)
            delta_v_rest = v_rest - pop.get("v_rest")[0]
            data.loc[(layer, ii), "v_rest"] = v_rest
            data.loc[(layer, ii), "v_rest_std"] = v_rest_std
            data.loc[(layer, ii), "delta_v_rest"] = delta_v_rest

        hicann_cfg.analog.disable(aout)
        adc.freeHandle()

        wafer.disconnect()
        return data


class HWUnit(object):
    """

    """

class HWModel(object):

    def __init__(self, wafer, calib_path = ''):
        self.wafer = wafer
        self.calib_path = calib_path
        self.marocco = PyMarocco()

    def _configure_marocco(self):
        self.marocco.neuron_placement.default_neuron_size(4)
        self.marocco.neuron_placement.minimize_number_of_sending_repeaters(False)
        self.marocco.merger_routing.strategy(marocco.merger_routing.one_to_one)
        self.marocco.l1_address_assignment.strategy(
            marocco.l1_address_assignment.low_first
            # marocco.l1_address_assignment.high_first
            # marocco.l1_address_assignment.alternating
        )
        # marocco.l1_routing.algorithm(marocco.l1_routing.dijkstra)
        try:
            self.marocco.experiment.truncate_membrane_traces(False)
            self.marocco.experiment.truncate_spike_times(False)
        except AttributeError:
            pass

        self.marocco.bkg_gen_isi = 125
        self.marocco.pll_freq = 125e6
        self.marocco.backend = PyMarocco.Hardware
        self.marocco.verification = PyMarocco.Skip
        self.marocco.calib_backend = PyMarocco.XML
        self.marocco.calib_path = self.calib_path
        self.marocco.defects.path = self.calib_path
        self.marocco.defects.backend = Defects.XML
        self.marocco.default_wafer = C.Wafer(self.wafer)
        self.marocco.param_trafo.use_big_capacitors = True
        self.marocco.input_placement.consider_firing_rate(True)
        self.marocco.input_placement.bandwidth_utilization(0.8)

    def _set_sthal_params(self, gmax, gmax_div, I_pl=None, min_V_syntc=None):
        """
        Note: Who knows what this does, but it is of course a good idea to let the user deal with it.
        """
        wafer = self.runtime.wafer()
        for hicann in wafer.getAllocatedHicannCoordinates():
            fgs = wafer[hicann].floating_gates
            for ii in xrange(fgs.getNoProgrammingPasses()):
                cfg = fgs.getFGConfig(C.Enum(ii))
                cfg.fg_biasn = 0
                cfg.fg_bias = 0
                fgs.setFGConfig(C.Enum(ii), cfg)

            if I_pl is not None:
                for denmem in C.iter_all(C.NeuronOnHICANN):
                    fgs.setNeuron(denmem, pyhalbe.HICANN.neuron_parameter.I_pl, I_pl)

            if min_V_syntc is not None:
                for param in [pyhalbe.HICANN.neuron_parameter.V_syntcx,
                              pyhalbe.HICANN.neuron_parameter.V_syntci]:
                    for denmem in C.iter_all(C.NeuronOnHICANN):
                        val = fgs.getNeuron(denmem, param)
                        if val < min_V_syntc:
                            fgs.setNeuron(denmem, param, min_V_syntc)

            for block in C.iter_all(C.FGBlockOnHICANN):
                fgs.setShared(block, pyhalbe.HICANN.shared_parameter.V_gmax0, gmax)
                fgs.setShared(block, pyhalbe.HICANN.shared_parameter.V_gmax1, gmax)
                fgs.setShared(block, pyhalbe.HICANN.shared_parameter.V_gmax2, gmax)
                fgs.setShared(block, pyhalbe.HICANN.shared_parameter.V_gmax3, gmax)

            for block in C.iter_all(C.FGBlockOnHICANN):
                fgs.setShared(block, pyhalbe.HICANN.shared_parameter.V_dllres, 275)
                fgs.setShared(block, pyhalbe.HICANN.shared_parameter.V_ccas, 800)

            for driver in C.iter_all(C.SynapseDriverOnHICANN):
                for row in C.iter_all(C.RowOnSynapseDriver):
                    # sum of both divisor values gives actual divisor value
                    # wafer[hicann].synapses[driver][row].set_gmax_div(
                    #     C.left, min(gmax_div, 15))
                    # wafer[hicann].synapses[driver][row].set_gmax_div(
                    #     C.right, max(gmax_div - 15, 0))
                    wafer[hicann].synapses[driver][row].set_gmax_div(
                        C.left, gmax_div)
                    wafer[hicann].synapses[driver][row].set_gmax_div(
                        C.right, gmax_div)


    def build():
        """
        
        """
        self._configure_marocco()









class PyHMFLIFNetwork(LIFNetwork):
    def __init__(self, wafer, **kwargs):
        calib_path = kwargs.pop("calib_path", ".")
        super(PyHMFLIFNetwork, self).__init__(pynn, **kwargs)

        self.synapses = {}
        self.pristine_decoder_values = {}
        self.marocco = marocco = PyMarocco()
        marocco.neuron_placement.default_neuron_size(4)
        marocco.neuron_placement.minimize_number_of_sending_repeaters(False)
        marocco.merger_routing.strategy(marocco.merger_routing.one_to_one)
        marocco.l1_address_assignment.strategy(
            marocco.l1_address_assignment.low_first
            # marocco.l1_address_assignment.high_first
            # marocco.l1_address_assignment.alternating
        )
        # marocco.l1_routing.algorithm(marocco.l1_routing.dijkstra)
        try:
            marocco.experiment.truncate_membrane_traces(False)
            marocco.experiment.truncate_spike_times(False)
        except AttributeError:
            pass

        marocco.bkg_gen_isi = 125
        marocco.pll_freq = 125e6
        marocco.backend = PyMarocco.Hardware
        marocco.verification = PyMarocco.Skip
        marocco.calib_backend = PyMarocco.XML
        marocco.calib_path = calib_path
        marocco.defects.path = calib_path
        marocco.defects.backend = Defects.XML
        marocco.default_wafer = C.Wafer(wafer)
        marocco.param_trafo.use_big_capacitors = True
        marocco.input_placement.consider_firing_rate(True)
        marocco.input_placement.bandwidth_utilization(0.8)

        self.runtime = None

    def setup(self, connections=True):
        self.pynn.end()
        self.runtime = runtime.Runtime(self.marocco.default_wafer)
        self.pynn.setup(marocco=self.marocco, marocco_runtime=self.runtime)
        return super(PyHMFLIFNetwork, self).setup(connections=connections)

    def set_sthal_params(self, gmax, gmax_div, I_pl=None, min_V_syntc=None):
        wafer = self.runtime.wafer()

        for hicann in wafer.getAllocatedHicannCoordinates():
            fgs = wafer[hicann].floating_gates
            for ii in xrange(fgs.getNoProgrammingPasses()):
                cfg = fgs.getFGConfig(C.Enum(ii))
                cfg.fg_biasn = 0
                cfg.fg_bias = 0
                fgs.setFGConfig(C.Enum(ii), cfg)

            if I_pl is not None:
                for denmem in C.iter_all(C.NeuronOnHICANN):
                    fgs.setNeuron(denmem, pyhalbe.HICANN.neuron_parameter.I_pl, I_pl)

            if min_V_syntc is not None:
                for param in [pyhalbe.HICANN.neuron_parameter.V_syntcx,
                              pyhalbe.HICANN.neuron_parameter.V_syntci]:
                    for denmem in C.iter_all(C.NeuronOnHICANN):
                        val = fgs.getNeuron(denmem, param)
                        if val < min_V_syntc:
                            fgs.setNeuron(denmem, param, min_V_syntc)

            for block in C.iter_all(C.FGBlockOnHICANN):
                fgs.setShared(block, pyhalbe.HICANN.shared_parameter.V_gmax0, gmax)
                fgs.setShared(block, pyhalbe.HICANN.shared_parameter.V_gmax1, gmax)
                fgs.setShared(block, pyhalbe.HICANN.shared_parameter.V_gmax2, gmax)
                fgs.setShared(block, pyhalbe.HICANN.shared_parameter.V_gmax3, gmax)

            for block in C.iter_all(C.FGBlockOnHICANN):
                fgs.setShared(block, pyhalbe.HICANN.shared_parameter.V_dllres, 275)
                fgs.setShared(block, pyhalbe.HICANN.shared_parameter.V_ccas, 800)

            for driver in C.iter_all(C.SynapseDriverOnHICANN):
                for row in C.iter_all(C.RowOnSynapseDriver):
                    # sum of both divisor values gives actual divisor value
                    # wafer[hicann].synapses[driver][row].set_gmax_div(
                    #     C.left, min(gmax_div, 15))
                    # wafer[hicann].synapses[driver][row].set_gmax_div(
                    #     C.right, max(gmax_div - 15, 0))
                    wafer[hicann].synapses[driver][row].set_gmax_div(
                        C.left, gmax_div)
                    wafer[hicann].synapses[driver][row].set_gmax_div(
                        C.right, gmax_div)

    def measure_resting_potential_prerun(self, duration_in_ms, **kwargs):
        for pop in sum(self.neurons, []):
            pop.set("v_thresh", 1e3)

        self.initial_config_run(duration_in_ms, **kwargs)

        wafer_cfg = self.runtime.wafer()
        wafer_cfg.connect(pysthal.MagicHardwareDatabase())

        indices = []
        by_hicann = {}
        placement = self.runtime.results().placement
        for layer, layer_neurons in enumerate(self.neurons):
            for ii, pop in enumerate(layer_neurons):
                pl_item, = placement.find(pop[0])
                nrn = pl_item.logical_neuron().front()
                hicann = nrn.toHICANNOnWafer()
                by_hicann.setdefault(hicann, []).append((layer, ii, pop, nrn))
                indices.append((layer, ii))

        index = pd.MultiIndex.from_tuples(indices, names=("layer", "neuron"))
        data = pd.DataFrame(dict(v_rest=np.zeros(len(indices))), index=index)

        for hicann, items in by_hicann.iteritems():
            self.log("info", "prerun on {}", hicann)
            hicann_cfg = wafer_cfg[hicann]
            aout = C.AnalogOnHICANN(0)
            adc = hicann_cfg.analogRecorder(aout)
            adc.setRecordingTime(duration_in_ms * 1e-7)

            for layer, ii, pop, nrn in items:
                hicann_cfg.enable_aout(nrn, aout)
                wafer_cfg.configure(UpdateAnalogOutputConfigurator())
                adc.record()
                trace = adc.trace()
                voltage = (trace - 1.2) * 1e3 / 10.
                time = (adc.getTimestamps() - 20e-6) * 1e7
                membrane = np.vstack((time, voltage)).T
                self.log(
                    "debug", "mean trace: {} {} -> {} ({} samples)",
                    hicann, nrn, np.mean(trace), len(trace))
                hicann_cfg.disable_aout()

                v_rest = np.mean(voltage)
                v_rest_std = np.std(voltage)
                delta_v_rest = v_rest - pop.get("v_rest")[0]
                data.loc[(layer, ii), "v_rest"] = v_rest
                data.loc[(layer, ii), "v_rest_std"] = v_rest_std
                data.loc[(layer, ii), "delta_v_rest"] = delta_v_rest

            hicann_cfg.analog.disable(aout)
            adc.freeHandle()

        wafer_cfg.disconnect()

        return data

    def initial_config_run(self, duration_in_ms, configurator=None, **kwargs):
        self.marocco.skip_mapping = False
        self.marocco.backend = PyMarocco.None

        self.pynn.reset()
        self.pynn.run(duration_in_ms)

        self.set_sthal_params(**kwargs)
        self.marocco.skip_mapping = True
        self.marocco.backend = PyMarocco.Hardware
        self.marocco.hicann_configurator = configurator or PyMarocco.HICANNv4Configurator

        self.pynn.reset()
        self.pynn.run(duration_in_ms)

        self.marocco.hicann_configurator = getattr(
            PyMarocco, "ParallelHICANNNoResetNoFGConfigurator",
            PyMarocco.NoResetNoFGConfigurator)

    def extract_synapse_placement(self):
        projections = self.stimulus_projections[:]
        for target_projections in self.projections.values():
            for layer_projections in target_projections:
                projections.extend(layer_projections.values())

        wafer_cfg = self.runtime.wafer()
        self.synapses = {}
        self.pristine_decoder_values = {}
        for proj in projections:
            proj_items = self.runtime.results().synapse_routing.synapses().find(proj)
            if not len(proj_items):
                # synapse loss :(
                continue
            proj_item, = proj_items
            self.synapses[proj] = synapse = proj_item.hardware_synapse()
            self.pristine_decoder_values[synapse] = copy.copy(
                wafer_cfg[synapse.toHICANNOnWafer()].synapses[synapse].decoder)

    def run(self, duration_in_ms, spikes_only=False):
        configurator = self.marocco.hicann_configurator
        if spikes_only:
            self.marocco.hicann_configurator = getattr(
                PyMarocco, "SpikeOnlyConfigurator",
                PyMarocco.OnlyNeuronNoResetNoFGConfigurator)
        self.pynn.run(duration_in_ms)
        self.marocco.hicann_configurator = configurator

    def set_weight(self, proj, value):
        assert proj.size == 1
        value = abs(value)

        try:
            synapse = self.synapses[proj]
        except KeyError:
            # synapse loss :(
            return

        proxy = self.runtime.wafer()[synapse.toHICANNOnWafer()].synapses[synapse]

        decoder = copy.copy(self.pristine_decoder_values[synapse])
        if value == 0:
            decoder = SYNAPSE_DECODER_DISABLED_SYNAPSE

        weight = SynapseWeight(value)
        proxy.decoder = decoder
        proxy.weight = weight

    def set_spike_times(self, pop, spike_times):
        self.runtime.results().spike_times.set(pop[0], spike_times)

    @property
    def hardware_weights_params(self):
        return dict(resolution=16)
