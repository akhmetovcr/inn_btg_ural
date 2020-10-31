//import {fromLonLat} from 'ol/proj';

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
const schladming = [{{ velo_map.longitude }}, {{ velo_map.latitude }}]//[65.534328, 57.153033];  // !!! В обратном порядке
const schladmingWebMercator = ol.proj.fromLonLat(schladming);

// устанавливаем карте координаты центра и зум Тюмени
var view = new ol.View({
  center: schladmingWebMercator,
  zoom: 12
});
map.setView(view);











