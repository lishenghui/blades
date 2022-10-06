pushd ../docs || exit
  rm -rf ./build
  make html
  rm -rf ~/Desktop/bladesteam.github.io/*
  cp -rf ./build/html/* ~/Desktop/bladesteam.github.io/
popd || exit
pushd ~/Desktop/bladesteam.github.io/ || exit
  git add .
  git commit -m "update"
  git push
popd || exit
