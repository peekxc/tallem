import sys 
print(f"Current PYTHONPATH via sys: {sys.path}")

def test_can_import_landmark():
	from tallem.pbm import landmark
	assert landmark is not None
	assert landmark.__name__ == 'tallem.pbm.landmark'
	assert 'maxmin' in dir(landmark)

def test_can_import_fast_svd():
	from tallem.pbm import fast_svd
	assert fast_svd is not None
	assert fast_svd.__name__ == 'tallem.pbm.fast_svd'
	assert 'StiefelLoss' in dir(fast_svd)

def test_can_import_tallem():
	import tallem
	assert tallem is not None
	assert tallem.__name__ == 'tallem'
	assert 'TALLEM' in dir(tallem)