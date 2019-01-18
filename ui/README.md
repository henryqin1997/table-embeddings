Web Table Visualization

* Install http-server
```
npm install -g http-server
```

* Change to project directory
```
cd table-embeddings
```

* Run the local file server
```
http-server -p 3000 --cors
```

* Run the visualization UI
```
http-server ui -p 8000
```

* Visualize a table in browser
```
open 'http://127.0.0.1:8000?file=data/train_100_sample/0/1438042988061.16_20150728002308-00010-ip-10-236-191-2_227851886_4.json'
```
