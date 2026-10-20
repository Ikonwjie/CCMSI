from data_provider.data_loader import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
    Dataset_Multivariate,
)
from torch.utils.data import DataLoader

data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "ECL": Dataset_Custom,
    "Traffic": Dataset_Custom,
    "Weather": Dataset_Custom,
    "exchange": Dataset_Custom,
    "custom": Dataset_Custom,
}

data_dict_multivariate = {
    "ETTh1": Dataset_Multivariate,
    "ETTh2": Dataset_Multivariate,
    "ETTm1": Dataset_Multivariate,
    "ETTm2": Dataset_Multivariate,
    "ECL": Dataset_Multivariate,
    "Traffic": Dataset_Multivariate,
    "Weather": Dataset_Multivariate,
    "exchange": Dataset_Multivariate,
    "custom": Dataset_Multivariate,
}


def data_provider(args, flag, multivariate=False):
    if multivariate:
        Data = data_dict_multivariate.get(args.data, Dataset_Multivariate)
    else:
        Data = data_dict.get(args.data, Dataset_Custom)
    
    timeenc = 0 if args.embed != "timeF" else 1
    percent = args.percent if hasattr(args, 'percent') else 100
    
    use_pregenerated = getattr(args, 'use_pregenerated', False)
    pregenerated_path = getattr(args, 'pregenerated_path', None)

    if flag == "test":
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    dataset_kwargs = dict(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        percent=percent,
        seasonal_patterns=args.seasonal_patterns if hasattr(args, 'seasonal_patterns') else None,
        test_noise_factor=getattr(args, "test_noise_factor", 0.0),
        test_noise_seed=getattr(args, "test_noise_seed", 2025),
    )
    
    if multivariate:
        dataset_kwargs['use_pregenerated'] = use_pregenerated
        dataset_kwargs['pregenerated_path'] = pregenerated_path
    
    data_set = Data(**dataset_kwargs)
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    
    return data_set, data_loader
