German_Credit_schema = {
    'status': {
        'range': {
            'min': 1.0,
            'max': 4.0
        },
        'dtype': float,
    },
    'credit_history': {
        'range': {
            'min': 0.0,
            'mean': 2.6
        },
        'dtype': float,
    },
    'amount': {
        'range': {
            'min': 5.0,
            'std': 1.0
        },
        'dtype': float,
    },
    'savings': {
        'range': {
            'min': 0.1,
            'std': 0.5
        },
        'dtype': float,
    }
}