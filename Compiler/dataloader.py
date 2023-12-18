from typing import Any, Callable, Iterable, TypeVar, Generic, List, Optional, Union
from tensor import *
import tensor as TS


class DataLoader():
    def __init__(self, samples, labels, batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = None, drop_last: bool = False):
        assert isinstance(samples, Array) or isinstance(samples, MultiArray) or isinstance(samples, Tensor)
        assert isinstance(labels, Array) or isinstance(labels, MultiArray) or isinstance(labels, Tensor)
        assert len(samples) == len(labels)
        
        self.size = len(samples)
        self.batch_size = batch_size
        
        batch_number = self.size//self.batch_size
        
        if not drop_last and self.size%self.batch_size != 0:
                batch_number = self.size//self.batch_size + 1
        
        tmp_samples = samples
        tmp_labels = labels
        if isinstance(samples, Tensor):
            tmp_samples = samples.value    
        
        if isinstance(labels, Tensor):
            tmp_labels = labels.value
            
        new_sample_shape = [batch_number, batch_size]
        for x in tmp_samples.sizes[1:]:
            new_sample_shape.append(x)
        new_label_shape = [batch_number, batch_size]
        for x in tmp_labels.sizes[1:]:
            new_label_shape.append(x)
            
        self.samples = Tensor(MultiArray(new_sample_shape, sfix))
        self.labels = Tensor(MultiArray(new_label_shape, sfix))
        self.data_buffer = Tensor(MultiArray(new_sample_shape[1:], sfix))
        self.label_buffer = Tensor(MultiArray(new_label_shape[1:], sfix))
        indices = regint.Array(len(samples))
        indices.assign(regint.inc(len(samples)))
        if shuffle:
            indices.shuffle()
        
        length = len(indices)-self.size%batch_size if drop_last else len(indices)

  
        @for_range(length)
        def _(i):
            self.samples[i//batch_size][i%batch_size] = tmp_samples[indices.get_vector(i, 1)].get_vector()
            self.labels[i//batch_size][i%batch_size] = tmp_labels[indices.get_vector(i, 1)].get_vector()
            

        if not drop_last:
            for i in range(batch_size - self.size%batch_size):
                self.samples[-1][self.size%batch_size+i]= self.samples.value[0][i].get_vector()
                self.labels[-1][self.size%batch_size+i] = self.labels.value[0][i].get_vector()
        self.size = self.samples.sizes[0]
        indices.delete()

    def get_size(self):
        return self.data_buffer
    
    def get_labelsize(self):
        return self.label_buffer
    
    def get_data(self, i):
        library.runtime_error_if(i >= len(self.samples),
                                    'dataset obatin overflow: %s/%s',
                                    i, len(self.samples))
        self.data_buffer.value[:] = self.samples[i].value[:]
        self.label_buffer.value[:] = self.labels[i].value[:]
        return self.data_buffer, self.label_buffer
    
    def __getitem__(self, i):
        return self.get_data(i)
               
        
        
        
        