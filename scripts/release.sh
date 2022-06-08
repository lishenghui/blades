pushd ../src
  rm -rf build dist blades.egg-info
  python3 setup.py sdist bdist_wheel
  python3 -m twine upload dist/*
#  git commit -m "update"
#  git push
popd