.PHONY : test sdist upload clean dist

test :
	pytest --cov pypsa --cov-report term-missing

sdist :
	python setup.py sdist

upload :
	twine upload dist/*

clean :
	rm dist/*

dist : sdist upload clean
