.PHONY: help clean clean-pyc clean-build list test test-all coverage docs release sdist

help:
	@echo
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc   - remove Python file artifacts"
	@echo "lint        - check style with flake8"
	@echo "test        - run tests quickly with the default Python"
	@echo "test-all    - run tests on every Python version with tox"
	@echo "coverage    - check code coverage quickly with the default Python"
	@echo "docs        - generate Sphinx HTML documentation, including API docs"
	@echo "sdist       - package"

clean: clean-build clean-pyc

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

lint:
	pylint --disable R0913,R0914 --msg-template='{msg_id}:{line:3d},{column}: {obj}: {msg}' UFalcon tests
	flake8 UFalcon tests

test:
	py.test tests

test-all:
	tox

coverage:
	coverage run --source UFalcon -m pytest tests
	coverage report -m
	coverage html
	open htmlcov/index.html

docs:
	rm -f docs/UFalcon.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ UFalcon
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	coverage run --source UFalcon -m pytest tests
	coverage report -m
	coverage html
	cp -R htmlcov docs/_build/html
	open docs/_build/html/index.html

sdist: clean
	pip freeze > requirements.rst
	python setup.py sdist
	ls -l dist
