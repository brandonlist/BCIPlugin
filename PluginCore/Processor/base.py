from collections.abc import Iterable
from functools import partial
import mne

class DumbSet():
    def __init__(self):
        self.raw = None
        self.windows = None

class ArrayPreprocessMaker():
    def __init__(self):
        self.datasets = []
        self.datasets.append(DumbSet())

    def make_array_processer(self,data,info):
        if data.ndim==3:
            self.datasets[0].windows = mne.EpochsArray(data=data,info=info)
        elif data.ndim==2:
            self.datasets[0].raw = mne.io.RawArray(data=data,info=info)
        return self

    def get_data(self):
        assert (self.datasets[0].raw is None and self.datasets[0].windows is not None) or (self.datasets[0].raw is not None and self.datasets[0].windows is None)
        if self.datasets[0].raw is not None:
            return self.datasets[0].raw.get_data()
        if self.datasets[0].windows is not None:
            return self.datasets[0].windows.get_data()

class Processor(object):
    """Preprocessor for an MNE Raw or Epochs object.

    Applies the provided preprocessing function to the data of a Raw or Epochs
    object.
    If the function is provided as a string, the method with that name will be
    used (e.g., 'pick_channels', 'filter', etc.).
    If it is provided as a callable and `apply_on_array` is True, the
    `apply_function` method of Raw and Epochs object will be used to apply the
    function on the internal arrays of Raw and Epochs.
    If `apply_on_array` is False, the callable must directly modify the Raw or
    Epochs object (e.g., by calling its method(s) or directly moraw_timepoint

    Parameters
    ----------
    fn: str or callable
        If str, the Raw/Epochs object must have a method with that name.
        If callable, directly apply the callable to the object.
    apply_on_array : bool
        Ignored if `fn` is not a callable. If True, the `apply_function` of Raw
        and Epochs object will be used to run `fn` on the underlying arrays
        directly. If False, `fn` must directly modify the Raw or Epochs object.
    kwargs:
        Keyword arguments to be forwarded to the MNE function.
    """
    def __init__(self, fn, apply_on_array=True, **kwargs):
        if callable(fn) and apply_on_array:
            channel_wise = kwargs.pop('channel_wise', False)
            kwargs = dict(fun=partial(fn, **kwargs), channel_wise=channel_wise)
            fn = 'apply_function'
        self.fn = fn
        self.kwargs = kwargs

    def apply(self, raw_or_epochs):
        try:
            self._try_apply(raw_or_epochs)
        except RuntimeError:
            # Maybe the function needs the data to be loaded and the data was
            # not loaded yet. Not all MNE functions need data to be loaded,
            # most importantly the 'crop' function can be lazily applied
            # without preloading data which can make the overall preprocessing
            # pipeline substantially faster.
            raw_or_epochs.load_data()
            self._try_apply(raw_or_epochs)

    def _try_apply(self, raw_or_epochs):
        print(raw_or_epochs)
        if callable(self.fn):
            self.fn(raw_or_epochs, **self.kwargs)
        else:
            if not hasattr(raw_or_epochs, self.fn):
                raise AttributeError(
                    f'MNE object does not have a {self.fn} method.')
            getattr(raw_or_epochs, self.fn)(**self.kwargs)

def preprocess(concat_ds, preprocessors, apply_on_array=False,info=None):
    """Apply preprocessors to a concat dataset or array data

    Parameters
    ----------
    concat_ds: BaseConcatDataset / numpy.ndarray
        A concat of BaseDataset or WindowsDataset datasets to be preprocessed. / array of data to be preprocessed, of course info should be provided
    preprocessors: list(Preprocessor)
        List of Preprocessor objects to apply to the dataset.

    Returns
    -------
    BaseConcatDataset:
        Preprocessed dataset.
    """
    if not isinstance(preprocessors, Iterable):
        raise ValueError(
            'preprocessors must be a list of Preprocessor objects.')
    for elem in preprocessors:
        assert hasattr(elem, 'apply'), (
            'Preprocessor object needs an `apply` method.')

    if apply_on_array:
        apm = ArrayPreprocessMaker()
        concat_ds = apm.make_array_processer(data=concat_ds, info=info)
    for ds in concat_ds.datasets:
        if hasattr(ds, 'raw'):
            if ds.raw is not None:
                _preprocess(ds.raw, preprocessors)
        elif hasattr(ds, 'windows'):
            if ds.windows is not None:
                _preprocess(ds.windows, preprocessors)
        else:
            raise ValueError(
                'Can only preprocess concatenation of BaseDataset or '
                'WindowsDataset, with either a `raw` or `windows` attribute.')
    if apply_on_array:
        return apm.get_data()

    # Recompute cumulative sizes as the transforms might have changed them
    # XXX: Ultimately, the best solution would be to have cumulative_size be
    #      a property of BaseConcatDataset.

    concat_ds.cumulative_sizes = concat_ds.cumsum(concat_ds.datasets)

def _preprocess(raw_or_epochs, preprocessors):
    """Apply preprocessor(s) to Raw or Epochs object.

    Parameters
    ----------
    raw_or_epochs: mne.io.Raw or mne.Epochs
        Object to preprocess.
    preprocessors: list(Preprocessor)
        List of preprocessors to apply to the dataset.
    """
    for preproc in preprocessors:
        preproc.apply(raw_or_epochs)
