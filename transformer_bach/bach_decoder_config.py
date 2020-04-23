from pathlib import Path

config = {
    'dataset':                     'bach',

    # --- Dataloader ---
    'dataloader_generator_kwargs': dict(
        include_transpositions=True,
        sequences_size=24
    ),

    # --- DataProcessor ---
    'data_processor_type':         'bach',
    'data_processor_kwargs':       dict(
        embedding_size=32
    ),

    # --- Decoder ---
    'decoder_type':                'transformer_relative',
    'decoder_kwargs':              dict(
        d_model=512,
        n_head=8,
        num_encoder_layers=4,
        num_decoder_layers=8,
        dim_feedforward=2048,
        positional_embedding_size=8,
        dropout=0.1,
    ),
    # ======== Training ========
    'lr':                          1e-5,
    'batch_size':                  8,
    'num_batches':                 2048,
    'num_epochs':                  10,

    # ======== model ID ========
    'timestamp':                   None,
    'savename':                    Path(__file__).stem,

    # ======== Update ========
    'update_iterations':           10,
    'generations_per_iteration':   50,
    'update_epochs':               100,
    'update_lr':                   1e-5,
}
