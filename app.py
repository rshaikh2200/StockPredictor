def get_available_models():
    """Get list of tickers that have trained models (paged through all R2 objects)."""
    tickers = []
    continuation_token = None

    while True:
        # on first call, token is None
        if continuation_token:
            resp = r2.list_objects_v2(
                Bucket=R2_BUCKET,
                Prefix=R2_PREFIX + '/',
                ContinuationToken=continuation_token
            )
        else:
            resp = r2.list_objects_v2(
                Bucket=R2_BUCKET,
                Prefix=R2_PREFIX + '/'
            )

        contents = resp.get('Contents', [])
        for obj in contents:
            key = obj['Key']
            if key.endswith('.keras'):
                filename = key.rsplit('/', 1)[-1]
                ticker = filename.split('_model_')[0]
                if ticker not in tickers:
                    tickers.append(ticker)

        # if there are more pages, keep going
        if resp.get('IsTruncated'):  
            continuation_token = resp.get('NextContinuationToken')
        else:
            break

    return sorted(tickers)
