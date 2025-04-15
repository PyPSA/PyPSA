// draw graph in sidebar, change global to true if prefered
function draw_graph_sidebar(myChart, global=false) {
  draw_graph(myChart, global)
}

// draw graph in modal view
function draw_graph_modal(myChart, global=true) {
  draw_graph(myChart, global)
}

// Disable Button in Heading
// add graph button next to light/dark mode switch if activated, but before search
$('.md-search').before('<form class="md-header__option"> \
                          <label id="graph_button" class="md-header__button md-icon"> \
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 171 146"> \
                              <path d="M171,100 C171,106.075 166.075,111 160,111 C154.016,111 149.158,106.219 149.014,100.27 L114.105,83.503 C111.564,86.693 108.179,89.18 104.282,90.616 L108.698,124.651 C112.951,126.172 116,130.225 116,135 C116,141.075 111.075,146 105,146 C98.925,146 94,141.075 94,135 C94,131.233 95.896,127.912 98.781,125.93 L94.364,91.896 C82.94,90.82 74,81.206 74,69.5 C74,69.479 74.001,69.46 74.001,69.439 L53.719,64.759 C50.642,70.269 44.76,74 38,74 C36.07,74 34.215,73.689 32.472,73.127 L20.624,90.679 C21.499,92.256 22,94.068 22,96 C22,102.075 17.075,107 11,107 C4.925,107 0,102.075 0,96 C0,89.925 4.925,85 11,85 C11.452,85 11.895,85.035 12.332,85.089 L24.184,67.531 C21.574,64.407 20,60.389 20,56 C20,48.496 24.594,42.07 31.121,39.368 L29.111,21.279 C24.958,19.707 22,15.704 22,11 C22,4.925 26.925,0 33,0 C39.075,0 44,4.925 44,11 C44,14.838 42.031,18.214 39.051,20.182 L41.061,38.279 C49.223,39.681 55.49,46.564 55.95,55.011 L76.245,59.694 C79.889,52.181 87.589,47 96.5,47 C100.902,47 105.006,48.269 108.475,50.455 L131.538,27.391 C131.192,26.322 131,25.184 131,24 C131,17.925 135.925,13 142,13 C148.075,13 153,17.925 153,24 C153,30.075 148.075,35 142,35 C140.816,35 139.678,34.808 138.609,34.461 L115.546,57.525 C117.73,60.994 119,65.098 119,69.5 C119,71.216 118.802,72.884 118.438,74.49 L153.345,91.257 C155.193,89.847 157.495,89 160,89 C166.075,89 171,93.925 171,100"> \
                              </path> \
                            </svg> \
                          </label> \
                        </form>');

// add a div to html in which the graph will be drawn
function add_graph_div(params) {
  $('.md-sidebar--secondary').each(function() {
    $(this).contents().append('<div id="graph" class="graph"></div>');
  });
};

add_graph_div();

function init_graph(params) {
  var myChart = echarts.init(document.getElementById('graph'), null, {
    renderer: 'canvas',
    useDirtyRect: false
  });
  return myChart;
};

var myChart = init_graph();

function draw_graph(myChart, global=true) {
  var _option = $.extend(true, {}, option);
  if(!global) {
    _option.series[0].data = graph_nodes();
    _option.series[0].links = graph_links();
  }
  // draw the graph
  myChart.setOption(_option);

  // add click event for nodes
  myChart.on('click', function (params) {
    if(params.dataType == "node") {
      window.location = params.value;
    }
  });

  // redraw on resize
  window.addEventListener('resize', myChart.resize);
};

var option;

function graph_links() {
  id = option.series[0].data.find(it => it.value === window.location.pathname).id;
  return option.series[0].links.filter(it => it.source === id || it.target === id);
}

function graph_nodes() {
  id = option.series[0].data.find(it => it.value === window.location.pathname).id;
  links = option.series[0].links.filter(it => it.source === id || it.target === id);
  ids = [];
  links.forEach(function (link) {
    ids.push(link.source, link.target);
  });
  return option.series[0].data.filter(it => [...new Set(ids)].includes(it.id));
}

$.getJSON(document.currentScript.src + '/../graph.json', function (graph) {
  myChart.hideLoading();

  // an offset of 5, so the dot/node is not that small
  graph.nodes.forEach(function (node) {
    node.symbolSize += 5;
  });

  // special feature, if u want to have long note titles, u can use this ' •'
  // to cut everything behind in graph view
  graph.nodes.forEach(function (node) {
    node.name = node.name.split(' •')[0];
  });
  graph.links.forEach(function (link) {
    link.source = link.source.split(' •')[0];
    link.target = link.target.split(' •')[0];
  });

  option = {
    tooltip: {
      show: false,
    },
    legend: [ // categories not supported yet
      //{
      //  data: graph.categories.map(function (a) {
      //    return a.name;
      //  })
      //}
    ],
    darkMode: "auto",
    backgroundColor: $("body").css("background-color"),
    series: [
      {
        name: 'Interactive Graph',
        type: 'graph',
        layout: 'force',
        data: graph.nodes,
        links: graph.links,
        categories: [],
        zoom: 2,
        roam: true,
        draggable: false,
        itemStyle: {
        color: '#d10a49'  // Change this to any color you want (blue in this example)
      },
        label: {
          show: true,
          position: 'right',
          formatter: '{b}'
        },
        emphasis: {
          focus: 'adjacency', // gray out not related nodes on mouse over
          label: {
            fontWeight: "bold"
	  }
	},
        labelLayout: {
          hideOverlap: false // true could be a good idea for large graphs
        },
        scaleLimit: {
          min: 0.5,
          max: 5
        },
        lineStyle: {
          color: 'source',
          curveness: 0 // 0.3, if not 0, link an backlink will have 2 lines
        }
      }
    ]
  };
  // initial draw in sidebar
  draw_graph_sidebar(myChart);
});

$("#__palette_0").change(function(){
  option.backgroundColor = $("body").css("background-color");
  myChart.setOption(option);
});
$("#__palette_1").change(function(){
  option.backgroundColor = $("body").css("background-color");
  myChart.setOption(option);
});

$('#graph_button').on('click', function (params) {
  $("body").css({ overflow: "hidden", position: "fixed" });
  $('#graph').remove();
  $('<div id="modal_background"><div id="graph" class="modal_graph"></div></div>').appendTo('body');
  $('#modal_background').on('click', function (params) {
    if(params.target === this) {
      $("body").css({ overflow: "", position: "" });
      $('#graph').remove();
      $('#modal_background').remove();
      add_graph_div();
      myChart = init_graph();
      // re-draw sidebar, e.g. switch back from modal view
      draw_graph_sidebar(myChart);
    }
  });
  myChart = init_graph();
  draw_graph_modal(myChart);
});
