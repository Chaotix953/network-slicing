from traffic import URLLCTraffic, MMTCTraffic

def test_urllc_init():
    gen = URLLCTraffic(lambda_rate=1000.0)
    assert gen.lambda_rate == 1000.0

def test_mmtc_init():
    gen = MMTCTraffic(lambda_on=5000.0, lambda_off=100.0, p_on=0.1, p_off=0.3)
    assert gen.lambda_on == 5000.0
