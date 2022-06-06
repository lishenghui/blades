make html
cp -rf ./build/html/* ~/Desktop/bladesteam.github.io/

pushd ~/Desktop/bladesteam.github.io/
  git add .
  git commit -m "update"
  git push
popd