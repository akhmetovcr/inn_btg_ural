
// инициализируем карту и указываем, в какой DOM-элемент она будет загружаться
var map = new ol.Map({
  target: 'map'
});

// создаем и подключаем слой OpenStreetMap
var osmLayer = new ol.layer.Tile({
  source: new ol.source.OSM()
});

map.addLayer(osmLayer);

/*57.2662, 65.9653*/

// устанавливаем карте координаты центра и зум
var view = new ol.View({
  center: [ 4188426.7147939987, 7508764.236877314 ],
  zoom: 12
});
map.setView(view);











