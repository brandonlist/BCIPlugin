from PluginCore.Datasets.base import BaseConcatDataset,fetch_data_description,BaseDataset

class CustomDataset(BaseConcatDataset):
    def __init__(self, **kwargs):
        if(len(kwargs)==1) and 'loading_script' in kwargs:
            loading_script = kwargs['loading_script']
            local_env = {}
            exec(open(loading_script,'r',encoding='UTF-8').read(),local_env)
            data = local_env['data']
            self.classes_codes = local_env['classes_codes']

            all_base_ds = []
            raws, description = fetch_data_description(data)
            for raw, (_, row) in zip(raws, description.iterrows()):
                all_base_ds.append(BaseDataset(raw, row))
            BaseConcatDataset.__init__(self, all_base_ds)
