from pathlib import Path

config = {
    'dataset':                     'bach',
    'random_seed':                  0,

    # --- Dataloader ---
    'dataloader_generator_kwargs': dict(
        include_transpositions=True,
        sequences_size=24,
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

    # ======== Generation =======
    'generation_kwargs':           dict(
        temperature=0.9,
        top_p=0.8,
    ),

    # ======== model ID ========
    'timestamp':                   None,
    'savename':                    'aug-gen',
    'description':                 'changing top_p back to 0.8',

    # ======== Training ========
    # 'num_epochs':                  50,
    # 'lr':                          1e-5,
    # 'batch_size':                  8,
    # 'num_batches':                 2048,

    # ======== Augmentative Generation ========
    'num_epochs':                  40,
    'generations_per_epoch':       50,
    'lr':                          1e-5,
    'batch_size':                  8,
    'num_batches':                 2048,
}
