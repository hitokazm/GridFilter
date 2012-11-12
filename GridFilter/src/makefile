
default:
	python -m compileall -lf .
	chmod 711 *.pyc
	mv *.pyc ../compiled/
	python setup.py build_ext --inplace
	
	cp -f *.so ../compiled/ 