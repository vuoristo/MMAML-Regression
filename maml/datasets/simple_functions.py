import torch
import numpy as np

from maml.datasets.metadataset import Task


def generate_sinusoid_batch(amp_range, phase_range, input_range, num_samples,
                            batch_size, oracle, bias=0):
    amp = np.random.uniform(amp_range[0], amp_range[1], [batch_size])
    phase = np.random.uniform(phase_range[0], phase_range[1], [batch_size])
    outputs = np.zeros([batch_size, num_samples, 1])
    inputs = np.zeros([batch_size, num_samples, 1])
    for i in range(batch_size):
        inputs[i] = np.random.uniform(input_range[0], input_range[1],
                                      [num_samples, 1])
        outputs[i] = amp[i] * np.sin(inputs[i] - phase[i]) + bias

    if oracle:
        amps = np.ones_like(inputs) * amp.reshape(-1, 1, 1)
        phases = np.ones_like(inputs) * phase.reshape(-1, 1, 1)
        inputs = np.concatenate((inputs, amps, phases), axis=2)

    return inputs, outputs, amp, phase


def generate_linear_batch(slope_range, intersect_range, input_range,
                          num_samples, batch_size, oracle):
    slope = np.random.uniform(slope_range[0], slope_range[1], [batch_size])
    intersect = np.random.uniform(intersect_range[0], intersect_range[1],
                                  [batch_size])
    outputs = np.zeros([batch_size, num_samples, 1])
    inputs = np.zeros([batch_size, num_samples, 1])
    for i in range(batch_size):
        inputs[i] = np.random.uniform(input_range[0], input_range[1],
                                      [num_samples, 1])
        outputs[i] = inputs[i] * slope[i] + intersect[i]

    if oracle:
        slopes = np.ones_like(inputs) * slope.reshape(-1, 1, 1)
        intersects = np.ones_like(inputs) * intersect.reshape(-1, 1, 1)
        inputs = np.concatenate((inputs, slopes, intersects), axis=2)

    return inputs, outputs, slope, intersect


class SimpleFunctionDataset(object):
    def __init__(self, num_total_batches=200000, num_samples_per_function=5,
                 num_val_samples=5, meta_batch_size=75, oracle=False,
                 train=True, device='cpu', dtype=torch.float, **kwargs):
        self._num_total_batches = num_total_batches
        self._num_samples_per_function = num_samples_per_function
        self._num_val_samples = num_val_samples
        self._num_total_samples = num_samples_per_function
        self._meta_batch_size = meta_batch_size
        self._oracle = oracle
        self._train = train
        self._device = device
        self._dtype = dtype

    def _generate_batch(self):
        raise NotImplementedError('Subclass should implement _generate_batch')

    def __iter__(self):
        for batch in range(self._num_total_batches):
            inputs, outputs, infos = self._generate_batch()

            train_tasks = []
            val_tasks = []
            for task in range(self._meta_batch_size):
                task_inputs = torch.tensor(
                    inputs[task], device=self._device, dtype=self._dtype)
                task_outputs = torch.tensor(
                    outputs[task], device=self._device, dtype=self._dtype)
                task_infos = infos[task]
                train_task = Task(task_inputs[self._num_val_samples:],
                                  task_outputs[self._num_val_samples:],
                                  task_infos)
                train_tasks.append(train_task)
                val_task = Task(task_inputs[:self._num_val_samples],
                                task_outputs[:self._num_val_samples],
                                task_infos)
                val_tasks.append(val_task)
            yield train_tasks, val_tasks

class BiasedSinusoidMetaDataset(SimpleFunctionDataset):
    def __init__(self, amp_range=[0.1, 5.0], phase_range=[0, np.pi],
                 input_range=[-5.0, 5.0], bias=0, **kwargs):
        super(BiasedSinusoidMetaDataset, self).__init__(**kwargs)
        self._amp_range = amp_range
        self._phase_range = phase_range
        self._input_range = input_range
        self._bias = bias

        if self._oracle:
            self.input_size = 3
        else:
            self.input_size = 1
        self.output_size = 1

    def _generate_batch(self):
        inputs, outputs, amp, phase = generate_sinusoid_batch(
            amp_range=self._amp_range, phase_range=self._phase_range,
            input_range=self._input_range,
            num_samples=self._num_total_samples,
            batch_size=self._meta_batch_size, oracle=self._oracle, bias=self._bias)
        task_infos = [{'task_id': 0, 'amp': amp[i], 'phase': phase[i]}
                      for i in range(len(amp))]
        return inputs, outputs, task_infos

class SinusoidMetaDataset(SimpleFunctionDataset):
    def __init__(self, amp_range=[0.1, 5.0], phase_range=[0, np.pi],
                 input_range=[-5.0, 5.0], **kwargs):
        super(SinusoidMetaDataset, self).__init__(**kwargs)
        self._amp_range = amp_range
        self._phase_range = phase_range
        self._input_range = input_range

        if self._oracle:
            self.input_size = 3
        else:
            self.input_size = 1
        self.output_size = 1

    def _generate_batch(self):
        inputs, outputs, amp, phase = generate_sinusoid_batch(
            amp_range=self._amp_range, phase_range=self._phase_range,
            input_range=self._input_range,
            num_samples=self._num_total_samples,
            batch_size=self._meta_batch_size, oracle=self._oracle)
        task_infos = [{'task_id': 0, 'amp': amp[i], 'phase': phase[i]}
                      for i in range(len(amp))]
        return inputs, outputs, task_infos


class LinearMetaDataset(SimpleFunctionDataset):
    def __init__(self, slope_range=[-3.0, 3.0], intersect_range=[-3, 3],
                 input_range=[-5.0, 5.0], **kwargs):
        super(LinearMetaDataset, self).__init__(**kwargs)
        self._slope_range = slope_range
        self._intersect_range = intersect_range
        self._input_range = input_range

        if self._oracle:
            self.input_size = 3
        else:
            self.input_size = 1
        self.output_size = 1

    def _generate_batch(self):
        inputs, outputs, slope, intersect = generate_linear_batch(
            slope_range=self._slope_range,
            intersect_range=self._intersect_range,
            input_range=self._input_range,
            num_samples=self._num_total_samples,
            batch_size=self._meta_batch_size, oracle=self._oracle)
        task_infos = [{'task_id': 0, 'slope': slope[i], 'intersect': intersect[i]}
                      for i in range(len(slope))]
        return inputs, outputs, task_infos

def generate_quadratic_batch(center_range, bias_range, sign_range, input_range,
                          num_samples, batch_size, oracle):
    center = np.random.uniform(center_range[0], center_range[1], [batch_size])
    bias   = np.random.uniform(bias_range[0], bias_range[1], [batch_size])

    # alpha range
    alpha  = np.random.uniform(sign_range[0], sign_range[1], [batch_size])
    sign   = np.random.randint(2, size=[batch_size])
    sign[sign == 0] = -1
    sign   = alpha * sign

    outputs = np.zeros([batch_size, num_samples, 1])
    inputs = np.zeros([batch_size, num_samples, 1])
    for i in range(batch_size):
        inputs[i] = np.random.uniform(input_range[0], input_range[1],
                                      [num_samples, 1])
        outputs[i] = sign[i] * (inputs[i] - center[i])**2 + bias[i]

    if oracle:
        centers = np.ones_like(inputs) * center.reshape(-1, 1, 1)
        biases = np.ones_like(inputs) * bias.reshape(-1, 1, 1)
        inputs = np.concatenate((inputs, centers, biases), axis=2)

    return inputs, outputs, sign, center, bias


class QuadraticMetaDataset(SimpleFunctionDataset):
    """ Quadratic function like: sign * (x - center)^2 + bias
    """
    def __init__(self, center_range=[-3.0, 3.0], bias_range=[-3, 3], sign_range=[0.02, 0.15],
                 input_range=[-5.0, 5.0], **kwargs):
        super(QuadraticMetaDataset, self).__init__(**kwargs)
        self._center_range = center_range
        self._bias_range = bias_range
        self._input_range = input_range
        self._sign_range = sign_range

        if self._oracle:
            self.input_size = 3
        else:
            self.input_size = 1
        self.output_size = 1

    def _generate_batch(self):
        inputs, outputs, sign, center, bias = generate_quadratic_batch(
            center_range=self._center_range,
            bias_range=self._bias_range,
            sign_range=self._sign_range,
            input_range=self._input_range,
            num_samples=self._num_total_samples,
            batch_size=self._meta_batch_size, oracle=self._oracle)
        task_infos = [{'task_id': 2, 'sign': sign[i], 'center': center[i], 'bias': bias[i]}
                      for i in range(len(sign))]
        return inputs, outputs, task_infos


class MixedFunctionsMetaDataset(SimpleFunctionDataset):
    def __init__(self, amp_range=[0.1, 5.0], phase_range=[0, np.pi],
                 input_range=[-5.0, 5.0], slope_range=[-3.0, 3.0],
                 intersect_range=[-3.0, 3.0], task_oracle=False,
                 noise_std=0, **kwargs):
        super(MixedFunctionsMetaDataset, self).__init__(**kwargs)
        self._amp_range = amp_range
        self._phase_range = phase_range
        self._slope_range = slope_range
        self._intersect_range = intersect_range
        self._input_range = input_range
        self._task_oracle = task_oracle
        self._noise_std = noise_std

        if not self._oracle:
            if not self._task_oracle:
                self.input_size = 1
            else:
                self.input_size = 2
        else:
            if not self._task_oracle:
                self.input_size = 3
            else:
                self.input_size = 4

        self.output_size = 1
        self.num_tasks = 2

    def _generate_batch(self):
        half_batch_size = self._meta_batch_size // 2
        sin_inputs, sin_outputs, amp, phase = generate_sinusoid_batch(
            amp_range=self._amp_range, phase_range=self._phase_range,
            input_range=self._input_range,
            num_samples=self._num_total_samples,
            batch_size=half_batch_size, oracle=self._oracle)
        sin_task_infos = [{'task_id': 0, 'amp': amp[i], 'phase': phase[i]}
                          for i in range(len(amp))]
        if self._task_oracle:
            sin_inputs = np.concatenate(
                (sin_inputs, np.zeros(sin_inputs.shape[:2] + (1,))), axis=2)

        lin_inputs, lin_outputs, slope, intersect = generate_linear_batch(
            slope_range=self._slope_range,
            intersect_range=self._intersect_range,
            input_range=self._input_range,
            num_samples=self._num_total_samples,
            batch_size=half_batch_size, oracle=self._oracle)
        lin_task_infos = [{'task_id': 1, 'slope': slope[i], 'intersect': intersect[i]}
                          for i in range(len(slope))]
        if self._task_oracle:
            lin_inputs = np.concatenate(
                (lin_inputs, np.ones(lin_inputs.shape[:2] + (1,))), axis=2)
        inputs = np.concatenate((sin_inputs, lin_inputs))
        outputs = np.concatenate((sin_outputs, lin_outputs))

        if self._noise_std > 0:
            outputs = outputs + np.random.normal(scale=self._noise_std, size=outputs.shape)
        task_infos = sin_task_infos + lin_task_infos
        return inputs, outputs, task_infos

class ManyFunctionsMetaDataset(SimpleFunctionDataset):
    def __init__(self, amp_range=[0.1, 5.0], phase_range=[0, np.pi],
                 input_range=[-5.0, 5.0], slope_range=[-3.0, 3.0],
                 intersect_range=[-3.0, 3.0], center_range=[-3.0, 3.0],
                 bias_range=[-3.0, 3.0], sign_range=[0.02, 0.15], task_oracle=False,
                 noise_std=0, **kwargs):
        super(ManyFunctionsMetaDataset, self).__init__(**kwargs)
        self._amp_range = amp_range
        self._phase_range = phase_range
        self._slope_range = slope_range
        self._intersect_range = intersect_range
        self._input_range = input_range
        self._center_range = center_range
        self._bias_range = bias_range
        self._sign_range = sign_range
        self._task_oracle = task_oracle
        self._noise_std = noise_std

        if not self._oracle:
            if not self._task_oracle:
                self.input_size = 1
            else:
                self.input_size = 2
        else:
            if not self._task_oracle:
                self.input_size = 3
            else:
                self.input_size = 4

        self.output_size = 1
        self.num_tasks = 2

    def _generate_batch(self):
        half_batch_size = self._meta_batch_size // 3
        sin_inputs, sin_outputs, amp, phase = generate_sinusoid_batch(
            amp_range=self._amp_range, phase_range=self._phase_range,
            input_range=self._input_range,
            num_samples=self._num_total_samples,
            batch_size=half_batch_size, oracle=self._oracle)
        sin_task_infos = [{'task_id': 0, 'amp': amp[i], 'phase': phase[i]}
                          for i in range(len(amp))]
        if self._task_oracle:
            sin_inputs = np.concatenate(
                (sin_inputs, np.zeros(sin_inputs.shape[:2] + (1,))), axis=2)

        lin_inputs, lin_outputs, slope, intersect = generate_linear_batch(
            slope_range=self._slope_range,
            intersect_range=self._intersect_range,
            input_range=self._input_range,
            num_samples=self._num_total_samples,
            batch_size=half_batch_size, oracle=self._oracle)
        lin_task_infos = [{'task_id': 1, 'slope': slope[i], 'intersect': intersect[i]}
                          for i in range(len(slope))]
        if self._task_oracle:
            lin_inputs = np.concatenate(
                (lin_inputs, np.ones(lin_inputs.shape[:2] + (1,))), axis=2)

        qua_inputs, qua_outputs, sign, center, bias = generate_quadratic_batch(
            center_range=self._center_range,
            bias_range=self._bias_range,
            sign_range=self._sign_range,
            input_range=self._input_range,
            num_samples=self._num_total_samples,
            batch_size=half_batch_size, oracle=self._oracle)
        qua_task_infos = [{'task_id': 2, 'sign': sign[i], 'center': center[i], 'bias': bias[i]}
                      for i in range(len(sign))]

        if self._task_oracle:
          qua_inputs = np.concatenate(
              (qua_inputs, np.ones(qua_inputs.shape[:2] + (1,))), axis=2)

        inputs = np.concatenate((sin_inputs, lin_inputs, qua_inputs))
        outputs = np.concatenate((sin_outputs, lin_outputs, qua_outputs))

        if self._noise_std > 0:
            outputs = outputs + np.random.normal(scale=self._noise_std, size=outputs.shape)
        task_infos = sin_task_infos + lin_task_infos + qua_task_infos
        return inputs, outputs, task_infos


class MultiSinusoidsMetaDataset(SimpleFunctionDataset):
    def __init__(self, amp_range=[0.1, 5.0], phase_range=[0, np.pi], biases=(-5, 5),
                 input_range=[-5.0, 5.0], task_oracle=False,
                 noise_std=0, **kwargs):
        super(MultiSinusoidsMetaDataset, self).__init__(**kwargs)
        self._amp_range = amp_range
        self._phase_range = phase_range
        self._input_range = input_range
        self._task_oracle = task_oracle
        self._noise_std = noise_std
        self._biases = biases

        if not self._oracle:
            if not self._task_oracle:
                self.input_size = 1
            else:
                self.input_size = 2
        else:
            if not self._task_oracle:
                self.input_size = 3
            else:
                self.input_size = 4

        self.output_size = 1
        self.num_tasks = 2

    def _generate_batch(self):
        half_batch_size = self._meta_batch_size // 2
        sin1_inputs, sin1_outputs, amp1, phase1 = generate_sinusoid_batch(
            amp_range=self._amp_range, phase_range=self._phase_range,
            input_range=self._input_range,
            num_samples=self._num_total_samples,
            batch_size=half_batch_size, oracle=self._oracle, bias=self._biases[0])
        sin1_task_infos = [{'task_id': 0, 'amp': amp1[i], 'phase': phase1[i]}
                          for i in range(len(amp1))]
        if self._task_oracle:
            sin1_inputs = np.concatenate(
                (sin1_inputs, np.zeros(sin1_inputs.shape[:2] + (1,))), axis=2)

        sin2_inputs, sin2_outputs, amp2, phase2 = generate_sinusoid_batch(
            amp_range=self._amp_range, phase_range=self._phase_range,
            input_range=self._input_range,
            num_samples=self._num_total_samples,
            batch_size=half_batch_size, oracle=self._oracle, bias=self._biases[0])
        sin2_task_infos = [{'task_id': 1, 'amp': amp2[i], 'phase': phase2[i]}
                          for i in range(len(amp2))]
        if self._task_oracle:
            sin2_inputs = np.concatenate(
                (sin2_inputs, np.zeros(sin2_inputs.shape[:2] + (1,))), axis=2)

        inputs = np.concatenate((sin1_inputs, sin2_inputs))
        outputs = np.concatenate((sin2_outputs, sin2_outputs))

        if self._noise_std > 0:
            outputs = outputs + np.random.normal(scale=self._noise_std,
                                                 size=outputs.shape)
        task_infos = sin1_task_infos + sin2_task_infos
        return inputs, outputs, task_infos

def generate_tanh_batch(center_range, bias_range, slope_range, input_range,
                          num_samples, batch_size, oracle):
    center = np.random.uniform(center_range[0], center_range[1], [batch_size])
    bias   = np.random.uniform(bias_range[0], bias_range[1], [batch_size])

    # alpha range
    slope  = np.random.uniform(slope_range[0], slope_range[1], [batch_size])

    outputs = np.zeros([batch_size, num_samples, 1])
    inputs = np.zeros([batch_size, num_samples, 1])
    for i in range(batch_size):
        inputs[i] = np.random.uniform(input_range[0], input_range[1],
                                      [num_samples, 1])
        outputs[i] = slope[i] * np.tanh(inputs[i] - center[i]) + bias[i]

    if oracle:
        centers = np.ones_like(inputs) * center.reshape(-1, 1, 1)
        biases = np.ones_like(inputs) * bias.reshape(-1, 1, 1)
        inputs = np.concatenate((inputs, centers, biases), axis=2)

    return inputs, outputs, slope, center, bias

def generate_abs_batch(slope_range, center_range, bias_range, input_range,
                          num_samples, batch_size, oracle):
    slope = np.random.uniform(slope_range[0], slope_range[1], [batch_size])
    bias = np.random.uniform(bias_range[0], bias_range[1], [batch_size])
    center = np.random.uniform(center_range[0], center_range[1], [batch_size])

    outputs = np.zeros([batch_size, num_samples, 1])
    inputs = np.zeros([batch_size, num_samples, 1])
    for i in range(batch_size):
        inputs[i] = np.random.uniform(input_range[0], input_range[1],
                                      [num_samples, 1])
        outputs[i] = np.abs(inputs[i] - center[i]) * slope[i] + bias[i]

    if oracle:
        slopes = np.ones_like(inputs) * slope.reshape(-1, 1, 1)
        intersects = np.ones_like(inputs) * intersect.reshape(-1, 1, 1)
        inputs = np.concatenate((inputs, slopes, intersects), axis=2)

    return inputs, outputs, slope, center, bias


class FiveFunctionsMetaDataset(SimpleFunctionDataset):
    def __init__(self, amp_range=[0.1, 5.0], phase_range=[0, np.pi],
                 input_range=[-5.0, 5.0], slope_range=[-3.0, 3.0],
                 intersect_range=[-3.0, 3.0], center_range=[-3.0, 3.0],
                 bias_range=[-3.0, 3.0], sign_range=[0.02, 0.15], task_oracle=False,
                 noise_std=0, **kwargs):
        super(FiveFunctionsMetaDataset, self).__init__(**kwargs)
        self._amp_range = amp_range
        self._phase_range = phase_range
        self._slope_range = slope_range
        self._intersect_range = intersect_range
        self._input_range = input_range
        self._center_range = center_range
        self._bias_range = bias_range
        self._sign_range = sign_range
        self._task_oracle = task_oracle
        self._noise_std = noise_std

        assert self._task_oracle == False
        if not self._oracle:
            if not self._task_oracle:
                self.input_size = 1
            else:
                self.input_size = 2
        else:
            if not self._task_oracle:
                self.input_size = 3
            else:
                self.input_size = 4

        self.output_size = 1
        self.num_tasks = 2

    def _generate_batch(self):
        assert self._meta_batch_size % 5 == 0, 'Error size of meta batch.'
        half_batch_size = self._meta_batch_size // 5
        sin_inputs, sin_outputs, amp, phase = generate_sinusoid_batch(
            amp_range=self._amp_range, phase_range=self._phase_range,
            input_range=self._input_range,
            num_samples=self._num_total_samples,
            batch_size=half_batch_size, oracle=self._oracle)
        sin_task_infos = [{'task_id': 0, 'amp': amp[i], 'phase': phase[i]}
                          for i in range(len(amp))]

        lin_inputs, lin_outputs, slope, intersect = generate_linear_batch(
            slope_range=self._slope_range,
            intersect_range=self._intersect_range,
            input_range=self._input_range,
            num_samples=self._num_total_samples,
            batch_size=half_batch_size, oracle=self._oracle)
        lin_task_infos = [{'task_id': 1, 'slope': slope[i], 'intersect': intersect[i]}
                          for i in range(len(slope))]

        qua_inputs, qua_outputs, sign, center, bias = generate_quadratic_batch(
            center_range=self._center_range,
            bias_range=self._bias_range,
            sign_range=self._sign_range,
            input_range=self._input_range,
            num_samples=self._num_total_samples,
            batch_size=half_batch_size, oracle=self._oracle)
        qua_task_infos = [{'task_id': 2, 'sign': sign[i], 'center': center[i], 'bias': bias[i]}
                      for i in range(len(sign))]

        tanh_inputs, tanh_outputs, slope, center, bias = generate_tanh_batch(
            center_range=self._center_range,
            bias_range=self._bias_range,
            slope_range=self._slope_range,
            input_range=self._input_range,
            num_samples=self._num_total_samples,
            batch_size=half_batch_size, oracle=self._oracle)
        tanh_task_infos = [{'task_id': 3, 'slope': slope[i], 'center': center[i], 'bias': bias[i]}
                      for i in range(len(sign))]

        abs_inputs, abs_outputs, slope, center, bias = generate_abs_batch(
            slope_range=self._slope_range,
            center_range=self._center_range,
            bias_range=self._bias_range,
            input_range=self._input_range,
            num_samples=self._num_total_samples,
            batch_size=half_batch_size, oracle=self._oracle)
        abs_task_infos = [{'task_id': 4, 'slope': slope[i], 'center': center[i], 'bias': bias[i]}
                      for i in range(len(sign))]

        inputs = np.concatenate((sin_inputs, lin_inputs, qua_inputs, tanh_inputs, abs_inputs))
        outputs = np.concatenate((sin_outputs, lin_outputs, qua_outputs, tanh_outputs, abs_outputs))

        if self._noise_std > 0:
            outputs = outputs + np.random.normal(scale=self._noise_std, size=outputs.shape)
        task_infos = sin_task_infos + lin_task_infos + qua_task_infos + tanh_task_infos + abs_task_infos
        return inputs, outputs, task_infos
