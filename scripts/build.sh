pushd ../src
  python3 setup.py sdist bdist_wheel

  pushd dist
    pip uninstall -y byzantinefl
    pip install --force-reinstall byzantinefl-0.0.1-py3-none-any.whl
  popd
popd