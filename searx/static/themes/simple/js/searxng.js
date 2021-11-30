/**
 * @license
 * (C) Copyright Contributors to the SearXNG project.
 * (C) Copyright Contributors to the searx project (2014 - 2021).
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */
window.searxng = (function(w, d) {

  'use strict';

  // not invented here tookit with bugs fixed elsewhere
  // purposes : be just good enough and as small as possible

  // from https://plainjs.com/javascript/events/live-binding-event-handlers-14/
  if (w.Element) {
    (function(ElementPrototype) {
      ElementPrototype.matches = ElementPrototype.matches ||
      ElementPrototype.matchesSelector ||
      ElementPrototype.webkitMatchesSelector ||
      ElementPrototype.msMatchesSelector ||
      function(selector) {
        var node = this, nodes = (node.parentNode || node.document).querySelectorAll(selector), i = -1;
        while (nodes[++i] && nodes[i] != node);
        return !!nodes[i];
      };
    })(Element.prototype);
  }

  function callbackSafe(callback, el, e) {
    try {
      callback.call(el, e);
    } catch (exception) {
      console.log(exception);
    }
  }

  var searxng = window.searxng || {};

  searxng.on = function(obj, eventType, callback, useCapture) {
    useCapture = useCapture || false;
    if (typeof obj !== 'string') {
      // obj HTMLElement, HTMLDocument
      obj.addEventListener(eventType, callback, useCapture);
    } else {
      // obj is a selector
      d.addEventListener(eventType, function(e) {
        var el = e.target || e.srcElement, found = false;
        while (el && el.matches && el !== d && !(found = el.matches(obj))) el = el.parentElement;
        if (found) callbackSafe(callback, el, e);
      }, useCapture);
    }
  };

  searxng.ready = function(callback) {
    if (document.readyState != 'loading') {
      callback.call(w);
    } else {
      w.addEventListener('DOMContentLoaded', callback.bind(w));
    }
  };

  searxng.http = function(method, url) {
    var req = new XMLHttpRequest(),
    resolve = function() {},
    reject = function() {},
    promise = {
      then: function(callback) { resolve = callback; return promise; },
      catch: function(callback) { reject = callback; return promise; }
    };

    try {
      req.open(method, url, true);

      // On load
      req.onload = function() {
        if (req.status == 200) {
          resolve(req.response, req.responseType);
        } else {
          reject(Error(req.statusText));
        }
      };

      // Handle network errors
      req.onerror = function() {
        reject(Error("Network Error"));
      };

      req.onabort = function() {
        reject(Error("Transaction is aborted"));
      };

      // Make the request
      req.send();
    } catch (ex) {
      reject(ex);
    }

    return promise;
  };

  searxng.loadStyle = function(src) {
    var path = searxng.static_path + src,
    id = "style_" + src.replace('.', '_'),
    s = d.getElementById(id);
    if (s === null) {
      s = d.createElement('link');
      s.setAttribute('id', id);
      s.setAttribute('rel', 'stylesheet');
      s.setAttribute('type', 'text/css');
      s.setAttribute('href', path);
      d.body.appendChild(s);
    }
  };

  searxng.loadScript = function(src, callback) {
    var path = searxng.static_path + src,
    id = "script_" + src.replace('.', '_'),
    s = d.getElementById(id);
    if (s === null) {
      s = d.createElement('script');
      s.setAttribute('id', id);
      s.setAttribute('src', path);
      s.onload = callback;
      s.onerror = function() {
        s.setAttribute('error', '1');
      };
      d.body.appendChild(s);
    } else if (!s.hasAttribute('error')) {
      try {
        callback.apply(s, []);
      } catch (exception) {
        console.log(exception);
      }
    } else {
      console.log("callback not executed : script '" + path + "' not loaded.");
    }
  };

  searxng.insertBefore = function (newNode, referenceNode) {
    referenceNode.parentNode.insertBefore(newNode, referenceNode);
  };

  searxng.insertAfter = function(newNode, referenceNode) {
    referenceNode.parentNode.insertAfter(newNode, referenceNode.nextSibling);
  };  

  searxng.on('.close', 'click', function() {
    this.parentNode.classList.add('invisible');
  });
  
  return searxng;
})(window, document);
;/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*global searxng*/

searxng.ready(function() {

  function isElementInDetail(el) {
    while (el !== undefined) {
      if (el.classList.contains('detail')) {
        return true;
      }
      if (el.classList.contains('result')) {
        // we found a result, no need to go to the root of the document:
        // el is not inside a <div class="detail"> element
        return false;
      }
      el = el.parentNode;
    }
    return false;
  }

  function getResultElement(el) {
    while (el !== undefined) {
      if (el.classList.contains('result')) {
        return el;
      }
      el = el.parentNode;
    }
    return undefined;
  }

  function isImageResult(resultElement) {
    return resultElement && resultElement.classList.contains('result-images');
  }

  searxng.on('.result', 'click', function(e) {
    if (!isElementInDetail(e.target)) {
      highlightResult(this)(true);
      let resultElement = getResultElement(e.target);
      if (isImageResult(resultElement)) {
        e.preventDefault();
        searxng.selectImage(resultElement);
      }
    }
  });

  searxng.on('.result a', 'focus', function(e) {
    if (!isElementInDetail(e.target)) {
      let resultElement = getResultElement(e.target);
      if (resultElement && resultElement.getAttribute("data-vim-selected") === null) {
        highlightResult(resultElement)(true);
      }
      if (isImageResult(resultElement)) {
        searxng.selectImage(resultElement);
      }
    }
  }, true);

  var vimKeys = {
    27: {
      key: 'Escape',
      fun: removeFocus,
      des: 'remove focus from the focused input',
      cat: 'Control'
    },
    73: {
      key: 'i',
      fun: searchInputFocus,
      des: 'focus on the search input',
      cat: 'Control'
    },
    66: {
      key: 'b',
      fun: scrollPage(-window.innerHeight),
      des: 'scroll one page up',
      cat: 'Navigation'
    },
    70: {
      key: 'f',
      fun: scrollPage(window.innerHeight),
      des: 'scroll one page down',
      cat: 'Navigation'
    },
    85: {
      key: 'u',
      fun: scrollPage(-window.innerHeight / 2),
      des: 'scroll half a page up',
      cat: 'Navigation'
    },
    68: {
      key: 'd',
      fun: scrollPage(window.innerHeight / 2),
      des: 'scroll half a page down',
      cat: 'Navigation'
    },
    71: {
      key: 'g',
      fun: scrollPageTo(-document.body.scrollHeight, 'top'),
      des: 'scroll to the top of the page',
      cat: 'Navigation'
    },
    86: {
      key: 'v',
      fun: scrollPageTo(document.body.scrollHeight, 'bottom'),
      des: 'scroll to the bottom of the page',
      cat: 'Navigation'
    },
    75: {
      key: 'k',
      fun: highlightResult('up'),
      des: 'select previous search result',
      cat: 'Results'
    },
    74: {
      key: 'j',
      fun: highlightResult('down'),
      des: 'select next search result',
      cat: 'Results'
    },
    80: {
      key: 'p',
      fun: GoToPreviousPage(),
      des: 'go to previous page',
      cat: 'Results'
    },
    78: {
      key: 'n',
      fun: GoToNextPage(),
      des: 'go to next page',
      cat: 'Results'
    },
    79: {
      key: 'o',
      fun: openResult(false),
      des: 'open search result',
      cat: 'Results'
    },
    84: {
      key: 't',
      fun: openResult(true),
      des: 'open the result in a new tab',
      cat: 'Results'
    },
    82: {
      key: 'r',
      fun: reloadPage,
      des: 'reload page from the server',
      cat: 'Control'
    },
    72: {
      key: 'h',
      fun: toggleHelp,
      des: 'toggle help window',
      cat: 'Other'
    }
  };

  if (searxng.hotkeys) {
    searxng.on(document, "keydown", function(e) {
      // check for modifiers so we don't break browser's hotkeys
      if (Object.prototype.hasOwnProperty.call(vimKeys, e.keyCode) && !e.ctrlKey && !e.altKey && !e.shiftKey && !e.metaKey) {
        var tagName = e.target.tagName.toLowerCase();
        if (e.keyCode === 27) {
          vimKeys[e.keyCode].fun(e);
        } else {
          if (e.target === document.body || tagName === 'a' || tagName === 'button') {
            e.preventDefault();
            vimKeys[e.keyCode].fun();
          }
        }
      }
    });
  }

  function highlightResult(which) {
    return function(noScroll) {
      var current = document.querySelector('.result[data-vim-selected]'),
      effectiveWhich = which;
      if (current === null) {
        // no selection : choose the first one
        current = document.querySelector('.result');
        if (current === null) {
          // no first one : there are no results
          return;
        }
        // replace up/down actions by selecting first one
        if (which === "down" || which === "up") {
          effectiveWhich = current;
        }
      }

      var next, results = document.querySelectorAll('.result');

      if (typeof effectiveWhich !== 'string') {
        next = effectiveWhich;
      } else {
        switch (effectiveWhich) {
          case 'visible':
          var top = document.documentElement.scrollTop || document.body.scrollTop;
          var bot = top + document.documentElement.clientHeight;

          for (var i = 0; i < results.length; i++) {
            next = results[i];
            var etop = next.offsetTop;
            var ebot = etop + next.clientHeight;

            if ((ebot <= bot) && (etop > top)) {
              break;
            }
          }
          break;
          case 'down':
          next = current.nextElementSibling;
          if (next === null) {
            next = results[0];
          }
          break;
          case 'up':
          next = current.previousElementSibling;
          if (next === null) {
            next = results[results.length - 1];
          }
          break;
          case 'bottom':
          next = results[results.length - 1];
          break;
          case 'top':
          /* falls through */
          default:
          next = results[0];
        }
      }

      if (next) {
        current.removeAttribute('data-vim-selected');
        next.setAttribute('data-vim-selected', 'true');
        var link = next.querySelector('h3 a') || next.querySelector('a');
        if (link !== null) {
          link.focus();
        }
        if (!noScroll) {
          scrollPageToSelected();
        }
      }
    };
  }

  function reloadPage() {
    document.location.reload(true);
  }

  function removeFocus(e) {
    const tagName = e.target.tagName.toLowerCase();
    if (document.activeElement && (tagName === 'input' || tagName === 'select' || tagName === 'textarea')) {
      document.activeElement.blur();
    } else {
      searxng.closeDetail();
    }
  }

  function pageButtonClick(css_selector) {
    return function() {
      var button = document.querySelector(css_selector);
      if (button) {
        button.click();
      }
    };
  }

  function GoToNextPage() {
    return pageButtonClick('nav#pagination .next_page button[type="submit"]');
  }

  function GoToPreviousPage() {
    return pageButtonClick('nav#pagination .previous_page button[type="submit"]');
  }

  function scrollPageToSelected() {
    var sel = document.querySelector('.result[data-vim-selected]');
    if (sel === null) {
      return;
    }
    var wtop = document.documentElement.scrollTop || document.body.scrollTop,
    wheight = document.documentElement.clientHeight,
    etop = sel.offsetTop,
    ebot = etop + sel.clientHeight,
    offset = 120;
    // first element ?
    if ((sel.previousElementSibling === null) && (ebot < wheight)) {
      // set to the top of page if the first element
      // is fully included in the viewport
      window.scroll(window.scrollX, 0);
      return;
    }
    if (wtop > (etop - offset)) {
      window.scroll(window.scrollX, etop - offset);
    } else {
      var wbot = wtop + wheight;
      if (wbot < (ebot + offset)) {
        window.scroll(window.scrollX, ebot - wheight + offset);
      }
    }
  }

  function scrollPage(amount) {
    return function() {
      window.scrollBy(0, amount);
      highlightResult('visible')();
    };
  }

  function scrollPageTo(position, nav) {
    return function() {
      window.scrollTo(0, position);
      highlightResult(nav)();
    };
  }

  function searchInputFocus() {
    window.scrollTo(0, 0);
    document.querySelector('#q').focus();
  }

  function openResult(newTab) {
    return function() {
      var link = document.querySelector('.result[data-vim-selected] h3 a');
      if (link === null) {
        link = document.querySelector('.result[data-vim-selected] > a');
      }
      if (link !== null) {
        var url = link.getAttribute('href');
        if (newTab) {
          window.open(url);
        } else {
          window.location.href = url;
        }
      }
    };
  }

  function initHelpContent(divElement) {
    var categories = {};

    for (var k in vimKeys) {
      var key = vimKeys[k];
      categories[key.cat] = categories[key.cat] || [];
      categories[key.cat].push(key);
    }

    var sorted = Object.keys(categories).sort(function(a, b) {
      return categories[b].length - categories[a].length;
    });

    if (sorted.length === 0) {
      return;
    }

    var html = '<a href="#" class="close" aria-label="close" title="close">×</a>';
    html += '<h3>How to navigate searx with Vim-like hotkeys</h3>';
    html += '<table>';

    for (var i = 0; i < sorted.length; i++) {
      var cat = categories[sorted[i]];

      var lastCategory = i === (sorted.length - 1);
      var first = i % 2 === 0;

      if (first) {
        html += '<tr>';
      }
      html += '<td>';

      html += '<h4>' + cat[0].cat + '</h4>';
      html += '<ul class="list-unstyled">';

      for (var cj in cat) {
        html += '<li><kbd>' + cat[cj].key + '</kbd> ' + cat[cj].des + '</li>';
      }

      html += '</ul>';
      html += '</td>'; // col-sm-*

      if (!first || lastCategory) {
        html += '</tr>'; // row
      }
    }

    html += '</table>';

     divElement.innerHTML = html;
  }

  function toggleHelp() {
    var helpPanel = document.querySelector('#vim-hotkeys-help');
    if (helpPanel === undefined || helpPanel === null) {
       // first call
      helpPanel = document.createElement('div');
         helpPanel.id = 'vim-hotkeys-help';
        helpPanel.className='dialog-modal';
      initHelpContent(helpPanel);
			initHelpContent(helpPanel);					
      initHelpContent(helpPanel);
      var body = document.getElementsByTagName('body')[0];
      body.appendChild(helpPanel);
    } else {
       // togggle hidden
      helpPanel.classList.toggle('invisible');
      return;
    }
  }

  searxng.scrollPageToSelected = scrollPageToSelected;
  searxng.selectNext = highlightResult('down');
  searxng.selectPrevious = highlightResult('up');
});
;/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* global L */
(function (w, d, searxng) {
  'use strict';

  searxng.ready(function () {
    searxng.on('.searxng_init_map', 'click', function(event) {
      // no more request
      this.classList.remove("searxng_init_map");

      //
      var leaflet_target = this.dataset.leafletTarget;
      var map_lon = parseFloat(this.dataset.mapLon);
      var map_lat = parseFloat(this.dataset.mapLat);
      var map_zoom = parseFloat(this.dataset.mapZoom);
      var map_boundingbox = JSON.parse(this.dataset.mapBoundingbox);
      var map_geojson = JSON.parse(this.dataset.mapGeojson);

      searxng.loadStyle('css/leaflet.css');
      searxng.loadScript('js/leaflet.js', function() {
        var map_bounds = null;
        if(map_boundingbox) {
          var southWest = L.latLng(map_boundingbox[0], map_boundingbox[2]);
          var northEast = L.latLng(map_boundingbox[1], map_boundingbox[3]);
          map_bounds = L.latLngBounds(southWest, northEast);
        }

        // init map
        var map = L.map(leaflet_target);
        // create the tile layer with correct attribution
        var osmMapnikUrl='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
        var osmMapnikAttrib='Map data © <a href="https://openstreetmap.org">OpenStreetMap</a> contributors';
        var osmMapnik = new L.TileLayer(osmMapnikUrl, {minZoom: 1, maxZoom: 19, attribution: osmMapnikAttrib});
        var osmWikimediaUrl='https://maps.wikimedia.org/osm-intl/{z}/{x}/{y}.png';
        var osmWikimediaAttrib = 'Wikimedia maps | Maps data © <a href="https://openstreetmap.org">OpenStreetMap contributors</a>';
        var osmWikimedia = new L.TileLayer(osmWikimediaUrl, {minZoom: 1, maxZoom: 19, attribution: osmWikimediaAttrib});
        // init map view
        if(map_bounds) {
          // TODO hack: https://github.com/Leaflet/Leaflet/issues/2021
          // Still useful ?
          setTimeout(function () {
            map.fitBounds(map_bounds, {
              maxZoom:17
            });
          }, 0);
        } else if (map_lon && map_lat) {
          if(map_zoom) {
            map.setView(new L.latLng(map_lat, map_lon),map_zoom);
          } else {
            map.setView(new L.latLng(map_lat, map_lon),8);
          }
        }

        map.addLayer(osmMapnik);

        var baseLayers = {
          "OSM Mapnik": osmMapnik,
          "OSM Wikimedia": osmWikimedia,
        };

        L.control.layers(baseLayers).addTo(map);

        if(map_geojson) {
          L.geoJson(map_geojson).addTo(map);
        } /*else if(map_bounds) {
          L.rectangle(map_bounds, {color: "#ff7800", weight: 3, fill:false}).addTo(map);
        }*/
      });

      // this event occour only once per element
      event.preventDefault();
    });
  });
})(window, document, window.searxng);
;/* SPDX-License-Identifier: AGPL-3.0-or-later */
(function (w, d, searxng) {
    'use strict';

    searxng.ready(function() {
        let engine_descriptions = null;
        function load_engine_descriptions() {
            if (engine_descriptions == null) {
                searxng.http("GET", "engine_descriptions.json").then(function(content) {
                    engine_descriptions = JSON.parse(content);
                    for (const [engine_name, description] of Object.entries(engine_descriptions)) {
                        let elements = d.querySelectorAll('[data-engine-name="' + engine_name + '"] .engine-description');
                        for(const element of elements) {
                            let source = ' (<i>' + searxng.translations['Source'] + ':&nbsp;' + description[1] + '</i>)';
                            element.innerHTML = description[0] + source;
                        }
                    }
                });
            }
        }

        if (d.querySelector('body[class="preferences_endpoint"]')) {
            for(const el of d.querySelectorAll('[data-engine-name]')) {
                searxng.on(el, 'mouseenter', load_engine_descriptions);
            }
        }
    });
})(window, document, window.searxng);
;/* SPDX-License-Identifier: AGPL-3.0-or-later */
(function(w, d, searxng) {
  'use strict';

  searxng.ready(function() {
    searxng.image_thumbnail_layout = new searxng.ImageLayout('#urls', '#urls .result-images', 'img.image_thumbnail', 14, 6, 200);
    searxng.image_thumbnail_layout.watch();

    searxng.on('.btn-collapse', 'click', function() {
      var btnLabelCollapsed = this.getAttribute('data-btn-text-collapsed');
      var btnLabelNotCollapsed = this.getAttribute('data-btn-text-not-collapsed');
      var target = this.getAttribute('data-target');
      var targetElement = d.querySelector(target);
      var html = this.innerHTML;
      if (this.classList.contains('collapsed')) {
        html = html.replace(btnLabelCollapsed, btnLabelNotCollapsed);
      } else {
        html = html.replace(btnLabelNotCollapsed, btnLabelCollapsed);
      }
      this.innerHTML = html;
      this.classList.toggle('collapsed');
      targetElement.classList.toggle('invisible');
    });

    searxng.on('.media-loader', 'click', function() {
      var target = this.getAttribute('data-target');
      var iframe_load = d.querySelector(target + ' > iframe');
      var srctest = iframe_load.getAttribute('src');
      if (srctest === null || srctest === undefined || srctest === false) {
        iframe_load.setAttribute('src', iframe_load.getAttribute('data-src'));
      }
    });

    searxng.selectImage = function(resultElement) {
      /*eslint no-unused-vars: 0*/
      if (resultElement) {
        // load full size image in background
        const imgElement = resultElement.querySelector('.result-images-source img');
        const thumbnailElement = resultElement.querySelector('.image_thumbnail');
        const detailElement = resultElement.querySelector('.detail');
        if (imgElement) {
          const imgSrc = imgElement.getAttribute('data-src');
          if (imgSrc) {
            const loader = d.createElement('div');
            const imgLoader = new Image();

            loader.classList.add('loader');
            detailElement.appendChild(loader);

            imgLoader.onload = e => {
              imgElement.src = imgSrc;
              loader.remove();
            };
            imgLoader.onerror = e => {
              loader.remove();
            };
            imgLoader.src = imgSrc;
            imgElement.src = thumbnailElement.src;
            imgElement.removeAttribute('data-src');
          }
        }
      }
      d.getElementById('results').classList.add('image-detail-open');
      searxng.image_thumbnail_layout.align();
      searxng.scrollPageToSelected();
    }

    searxng.closeDetail = function(e) {
      d.getElementById('results').classList.remove('image-detail-open');
      searxng.image_thumbnail_layout.align();
      searxng.scrollPageToSelected();
    }
    searxng.on('.result-detail-close', 'click', e => { 
      e.preventDefault();
      searxng.closeDetail();
    });
    searxng.on('.result-detail-previous', 'click', e => searxng.selectPrevious(false));
    searxng.on('.result-detail-next', 'click', e => searxng.selectNext(false));

    w.addEventListener('scroll', function() {
      var e = d.getElementById('backToTop'),
      scrollTop = document.documentElement.scrollTop || document.body.scrollTop,
      results = d.getElementById('results');
      if (e !== null) {
        if (scrollTop >= 100) {
          results.classList.add('scrolling');
        } else {
          results.classList.remove('scrolling');
        }
      }
    }, true);

  });

})(window, document, window.searxng);
;/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* global AutoComplete */
(function(w, d, searxng) {
  'use strict';

  var firstFocus = true, qinput_id = "q", qinput;

  function placeCursorAtEnd(element) {
    if (element.setSelectionRange) {
      var len = element.value.length;
      element.setSelectionRange(len, len);
    }
  }

  function submitIfQuery() {
    if (qinput.value.length  > 0) {
      var search = document.getElementById('search');
      setTimeout(search.submit.bind(search), 0);
    }
  }

  function createClearButton(qinput) {
    var cs = document.getElementById('clear_search');
    var updateClearButton = function() {
      if (qinput.value.length === 0) {
	cs.classList.add("empty");
      } else {
	cs.classList.remove("empty");
      }
    };

    // update status, event listener
    updateClearButton();
    cs.addEventListener('click', function() {
      qinput.value='';
      qinput.focus();
      updateClearButton();
    });
    qinput.addEventListener('keyup', updateClearButton, false);
  }

  searxng.ready(function() {
    qinput = d.getElementById(qinput_id);

    function placeCursorAtEndOnce() {
      if (firstFocus) {
        placeCursorAtEnd(qinput);
        firstFocus = false;
      } else {
        // e.preventDefault();
      }
    }

    if (qinput !== null) {
      // clear button
      createClearButton(qinput);
      
      // autocompleter
      if (searxng.autocompleter) {
        searxng.autocomplete = AutoComplete.call(w, {
          Url: "./autocompleter",
          EmptyMessage: searxng.translations.no_item_found,
          HttpMethod: searxng.method,
          HttpHeaders: {
            "Content-type": "application/x-www-form-urlencoded",
            "X-Requested-With": "XMLHttpRequest"
          },
          MinChars: 4,
          Delay: 300,
        }, "#" + qinput_id);

        // hack, see : https://github.com/autocompletejs/autocomplete.js/issues/37
        w.addEventListener('resize', function() {
          var event = new CustomEvent("position");
          qinput.dispatchEvent(event);
        });
      }

      qinput.addEventListener('focus', placeCursorAtEndOnce, false);
      qinput.focus();
    }

    // vanilla js version of search_on_category_select.js
    if (qinput !== null && d.querySelector('.help') != null && searxng.search_on_category_select) {
      d.querySelector('.help').className='invisible';

      searxng.on('#categories input', 'change', function() {
        var i, categories = d.querySelectorAll('#categories input[type="checkbox"]');
        for(i=0; i<categories.length; i++) {
          if (categories[i] !== this && categories[i].checked) {
            categories[i].click();
          }
        }
        if (! this.checked) {
          this.click();
        }
        submitIfQuery();
        return false;
      });

      searxng.on(d.getElementById('safesearch'), 'change', submitIfQuery);
      searxng.on(d.getElementById('time_range'), 'change', submitIfQuery);
      searxng.on(d.getElementById('language'), 'change', submitIfQuery);
    }

  });

})(window, document, window.searxng);
;/**
*
* Google Image Layout v0.0.1
* Description, by Anh Trinh.
* Heavily modified for searx
* https://ptgamr.github.io/2014-09-12-google-image-layout/
* https://ptgamr.github.io/google-image-layout/src/google-image-layout.js
*
* @license Free to use under the MIT License.
*
* @example <caption>Example usage of searxng.ImageLayout class.</caption>
* searxng.image_thumbnail_layout = new searxng.ImageLayout(
*     '#urls',                 // container_selector
*     '#urls .result-images',  // results_selector
*     'img.image_thumbnail',   // img_selector
*     14,                      // verticalMargin
*     6,                       // horizontalMargin
*     200                      // maxHeight
* );
* searxng.image_thumbnail_layout.watch();
*/


(function (w, d) {
  function ImageLayout(container_selector, results_selector, img_selector, verticalMargin, horizontalMargin, maxHeight) {
    this.container_selector = container_selector;
    this.results_selector = results_selector;
    this.img_selector = img_selector;
    this.verticalMargin = verticalMargin;
    this.horizontalMargin = horizontalMargin;
    this.maxHeight = maxHeight;
    this.isAlignDone = true;
  }

  /**
  * Get the height that make all images fit the container
  *
  * width = w1 + w2 + w3 + ... = r1*h + r2*h + r3*h + ...
  *
  * @param  {[type]} images the images to be calculated
  * @param  {[type]} width  the container witdth
  * @param  {[type]} margin the margin between each image
  *
  * @return {[type]}        the height
  */
  ImageLayout.prototype._getHeigth = function (images, width) {
    var i, img;
    var r = 0;

    for (i = 0; i < images.length; i++) {
      img = images[i];
      if ((img.naturalWidth > 0) && (img.naturalHeight > 0)) {
        r += img.naturalWidth / img.naturalHeight;
      } else {
        // assume that not loaded images are square
        r += 1;
      }
    }

    return (width - images.length * this.verticalMargin) / r; //have to round down because Firefox will automatically roundup value with number of decimals > 3
  };

  ImageLayout.prototype._setSize = function (images, height) {
    var i, img, imgWidth;
    var imagesLength = images.length, resultNode;

    for (i = 0; i < imagesLength; i++) {
      img = images[i];
      if ((img.naturalWidth > 0) && (img.naturalHeight > 0)) {
        imgWidth = height * img.naturalWidth / img.naturalHeight;
      } else {
        // not loaded image : make it square as _getHeigth said it
        imgWidth = height;
      }
      img.style.width = imgWidth + 'px';
      img.style.height = height + 'px';
      img.style.marginLeft = this.horizontalMargin + 'px';
      img.style.marginTop = this.horizontalMargin + 'px';
      img.style.marginRight = this.verticalMargin - 7 + 'px'; // -4 is the negative margin of the inline element
      img.style.marginBottom = this.verticalMargin - 7 + 'px';
      resultNode = img.parentNode.parentNode;
      if (!resultNode.classList.contains('js')) {
        resultNode.classList.add('js');
      }
    }
  };

  ImageLayout.prototype._alignImgs = function (imgGroup) {
    var isSearching, slice, i, h;
    var containerElement = d.querySelector(this.container_selector);
    var containerCompStyles = window.getComputedStyle(containerElement);
    var containerPaddingLeft = parseInt(containerCompStyles.getPropertyValue('padding-left'), 10);
    var containerPaddingRight = parseInt(containerCompStyles.getPropertyValue('padding-right'), 10);
    var containerWidth = containerElement.clientWidth - containerPaddingLeft - containerPaddingRight;

    while (imgGroup.length > 0) {
      isSearching = true;
      for (i = 1; i <= imgGroup.length && isSearching; i++) {
        slice = imgGroup.slice(0, i);
        h = this._getHeigth(slice, containerWidth);
        if (h < this.maxHeight) {
          this._setSize(slice, h);
          // continue with the remaining images
          imgGroup = imgGroup.slice(i);
          isSearching = false;
        }
      }
      if (isSearching) {
        this._setSize(slice, Math.min(this.maxHeight, h));
        break;
      }
    }
  };

  ImageLayout.prototype.align = function () {
    var i;
    var results_selectorNode = d.querySelectorAll(this.results_selector);
    var results_length = results_selectorNode.length;
    var previous = null;
    var current = null;
    var imgGroup = [];

    for (i = 0; i < results_length; i++) {
      current = results_selectorNode[i];
      if (current.previousElementSibling !== previous && imgGroup.length > 0) {
        // the current image is not connected to previous one
        // so the current image is the start of a new group of images.
        // so call _alignImgs to align the current group
        this._alignImgs(imgGroup);
        // and start a new empty group of images
        imgGroup = [];
      }
      // add the current image to the group (only the img tag)
      imgGroup.push(current.querySelector(this.img_selector));
      // update the previous variable
      previous = current;
    }
    // align the remaining images
    if (imgGroup.length > 0) {
      this._alignImgs(imgGroup);
    }
  };

  ImageLayout.prototype.watch = function () {
    var i, img;
    var obj = this;
    var results_nodes = d.querySelectorAll(this.results_selector);
    var results_length = results_nodes.length;

    function img_load_error(event) {
      // console.log("ERROR can't load: " + event.originalTarget.src);
      event.originalTarget.src = w.searxng.static_path + w.searxng.theme.img_load_error;
    }

    function throttleAlign() {
      if (obj.isAlignDone) {
        obj.isAlignDone = false;
        setTimeout(function () {
          obj.align();
          obj.isAlignDone = true;
        }, 100);
      }
    }

    // https://developer.mozilla.org/en-US/docs/Web/API/Window/pageshow_event
    w.addEventListener('pageshow', throttleAlign);
    // https://developer.mozilla.org/en-US/docs/Web/API/FileReader/load_event
    w.addEventListener('load', throttleAlign);
    // https://developer.mozilla.org/en-US/docs/Web/API/Window/resize_event
    w.addEventListener('resize', throttleAlign);

    for (i = 0; i < results_length; i++) {
      img = results_nodes[i].querySelector(this.img_selector);
      if (img !== null && img !== undefined) {
        img.addEventListener('load', throttleAlign);
        // https://developer.mozilla.org/en-US/docs/Web/API/GlobalEventHandlers/onerror
        img.addEventListener('error', throttleAlign);
        if (w.searxng.theme.img_load_error) {
          img.addEventListener('error', img_load_error, {once: true});
        }
      }
    }
  };

  w.searxng.ImageLayout = ImageLayout;

}(window, document));
;(function(f){if(typeof exports==="object"&&typeof module!=="undefined"){module.exports=f()}else if(typeof define==="function"&&define.amd){define([],f)}else{var g;if(typeof window!=="undefined"){g=window}else if(typeof global!=="undefined"){g=global}else if(typeof self!=="undefined"){g=self}else{g=this}g.AutoComplete = f()}})(function(){var define,module,exports;return (function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
/*
 * @license MIT
 *
 * Autocomplete.js v2.7.1
 * Developed by Baptiste Donaux
 * http://autocomplete-js.com
 *
 * (c) 2017, Baptiste Donaux
 */
"use strict";
var ConditionOperator;
(function (ConditionOperator) {
    ConditionOperator[ConditionOperator["AND"] = 0] = "AND";
    ConditionOperator[ConditionOperator["OR"] = 1] = "OR";
})(ConditionOperator || (ConditionOperator = {}));
var EventType;
(function (EventType) {
    EventType[EventType["KEYDOWN"] = 0] = "KEYDOWN";
    EventType[EventType["KEYUP"] = 1] = "KEYUP";
})(EventType || (EventType = {}));
/**
 * Core
 *
 * @class
 * @author Baptiste Donaux <baptiste.donaux@gmail.com> @baptistedonaux
 */
var AutoComplete = /** @class */ (function () {
    // Constructor
    function AutoComplete(params, selector) {
        if (params === void 0) { params = {}; }
        if (selector === void 0) { selector = "[data-autocomplete]"; }
        if (Array.isArray(selector)) {
            selector.forEach(function (s) {
                new AutoComplete(params, s);
            });
        }
        else if (typeof selector == "string") {
            var elements = document.querySelectorAll(selector);
            Array.prototype.forEach.call(elements, function (input) {
                new AutoComplete(params, input);
            });
        }
        else {
            var specificParams = AutoComplete.merge(AutoComplete.defaults, params, {
                DOMResults: document.createElement("div")
            });
            AutoComplete.prototype.create(specificParams, selector);
            return specificParams;
        }
    }
    AutoComplete.prototype.create = function (params, element) {
        params.Input = element;
        if (params.Input.nodeName.match(/^INPUT$/i) && (params.Input.hasAttribute("type") === false || params.Input.getAttribute("type").match(/^TEXT|SEARCH$/i))) {
            params.Input.setAttribute("autocomplete", "off");
            params._Position(params);
            params.Input.parentNode.appendChild(params.DOMResults);
            params.$Listeners = {
                blur: params._Blur.bind(params),
                destroy: AutoComplete.prototype.destroy.bind(null, params),
                focus: params._Focus.bind(params),
                keyup: AutoComplete.prototype.event.bind(null, params, EventType.KEYUP),
                keydown: AutoComplete.prototype.event.bind(null, params, EventType.KEYDOWN),
                position: params._Position.bind(params)
            };
            for (var event in params.$Listeners) {
                params.Input.addEventListener(event, params.$Listeners[event]);
            }
        }
    };
    AutoComplete.prototype.getEventsByType = function (params, type) {
        var mappings = {};
        for (var key in params.KeyboardMappings) {
            var event = EventType.KEYUP;
            if (params.KeyboardMappings[key].Event !== undefined) {
                event = params.KeyboardMappings[key].Event;
            }
            if (event == type) {
                mappings[key] = params.KeyboardMappings[key];
            }
        }
        return mappings;
    };
    AutoComplete.prototype.event = function (params, type, event) {
        var eventIdentifier = function (condition) {
            if ((match === true && mapping.Operator == ConditionOperator.AND) || (match === false && mapping.Operator == ConditionOperator.OR)) {
                condition = AutoComplete.merge({
                    Not: false
                }, condition);
                if (condition.hasOwnProperty("Is")) {
                    if (condition.Is == event.keyCode) {
                        match = !condition.Not;
                    }
                    else {
                        match = condition.Not;
                    }
                }
                else if (condition.hasOwnProperty("From") && condition.hasOwnProperty("To")) {
                    if (event.keyCode >= condition.From && event.keyCode <= condition.To) {
                        match = !condition.Not;
                    }
                    else {
                        match = condition.Not;
                    }
                }
            }
        };
        for (var name in AutoComplete.prototype.getEventsByType(params, type)) {
            var mapping = AutoComplete.merge({
                Operator: ConditionOperator.AND
            }, params.KeyboardMappings[name]), match = ConditionOperator.AND == mapping.Operator;
            mapping.Conditions.forEach(eventIdentifier);
            if (match === true) {
                mapping.Callback.call(params, event);
            }
        }
    };
    AutoComplete.prototype.makeRequest = function (params, callback, callbackErr) {
        var propertyHttpHeaders = Object.getOwnPropertyNames(params.HttpHeaders), request = new XMLHttpRequest(), method = params._HttpMethod(), url = params._Url(), queryParams = params._Pre(), queryParamsStringify = encodeURIComponent(params._QueryArg()) + "=" + encodeURIComponent(queryParams);
        if (method.match(/^GET$/i)) {
            if (url.indexOf("?") !== -1) {
                url += "&" + queryParamsStringify;
            }
            else {
                url += "?" + queryParamsStringify;
            }
        }
        request.open(method, url, true);
        for (var i = propertyHttpHeaders.length - 1; i >= 0; i--) {
            request.setRequestHeader(propertyHttpHeaders[i], params.HttpHeaders[propertyHttpHeaders[i]]);
        }
        request.onreadystatechange = function () {
            if (request.readyState == 4 && request.status == 200) {
                params.$Cache[queryParams] = request.response;
                callback(request.response);
            }
            else if (request.status >= 400) {
                callbackErr();
            }
        };
        return request;
    };
    AutoComplete.prototype.ajax = function (params, request, timeout) {
        if (timeout === void 0) { timeout = true; }
        if (params.$AjaxTimer) {
            window.clearTimeout(params.$AjaxTimer);
        }
        if (timeout === true) {
            params.$AjaxTimer = window.setTimeout(AutoComplete.prototype.ajax.bind(null, params, request, false), params.Delay);
        }
        else {
            if (params.Request) {
                params.Request.abort();
            }
            params.Request = request;
            params.Request.send(params._QueryArg() + "=" + params._Pre());
        }
    };
    AutoComplete.prototype.cache = function (params, callback, callbackErr) {
        var response = params._Cache(params._Pre());
        if (response === undefined) {
            var request = AutoComplete.prototype.makeRequest(params, callback, callbackErr);
            AutoComplete.prototype.ajax(params, request);
        }
        else {
            callback(response);
        }
    };
    AutoComplete.prototype.destroy = function (params) {
        for (var event in params.$Listeners) {
            params.Input.removeEventListener(event, params.$Listeners[event]);
        }
        params.DOMResults.parentNode.removeChild(params.DOMResults);
    };
    AutoComplete.merge = function () {
        var merge = {}, tmp;
        for (var i = 0; i < arguments.length; i++) {
            for (tmp in arguments[i]) {
                merge[tmp] = arguments[i][tmp];
            }
        }
        return merge;
    };
    AutoComplete.defaults = {
        Delay: 150,
        EmptyMessage: "No result here",
        Highlight: {
            getRegex: function (value) {
                return new RegExp(value, "ig");
            },
            transform: function (value) {
                return "<strong>" + value + "</strong>";
            }
        },
        HttpHeaders: {
            "Content-type": "application/x-www-form-urlencoded"
        },
        Limit: 0,
        MinChars: 0,
        HttpMethod: "GET",
        QueryArg: "q",
        Url: null,
        KeyboardMappings: {
            "Enter": {
                Conditions: [{
                        Is: 13,
                        Not: false
                    }],
                Callback: function (event) {
                    if (this.DOMResults.getAttribute("class").indexOf("open") != -1) {
                        var liActive = this.DOMResults.querySelector("li.active");
                        if (liActive !== null) {
                            event.preventDefault();
                            this._Select(liActive);
                            this.DOMResults.setAttribute("class", "autocomplete");
                        }
                    }
                },
                Operator: ConditionOperator.AND,
                Event: EventType.KEYDOWN
            },
            "KeyUpAndDown_down": {
                Conditions: [{
                        Is: 38,
                        Not: false
                    },
                    {
                        Is: 40,
                        Not: false
                    }],
                Callback: function (event) {
                    event.preventDefault();
                },
                Operator: ConditionOperator.OR,
                Event: EventType.KEYDOWN
            },
            "KeyUpAndDown_up": {
                Conditions: [{
                        Is: 38,
                        Not: false
                    },
                    {
                        Is: 40,
                        Not: false
                    }],
                Callback: function (event) {
                    event.preventDefault();
                    var first = this.DOMResults.querySelector("li:first-child:not(.locked)"), last = this.DOMResults.querySelector("li:last-child:not(.locked)"), active = this.DOMResults.querySelector("li.active");
                    if (active) {
                        var currentIndex = Array.prototype.indexOf.call(active.parentNode.children, active), position = currentIndex + (event.keyCode - 39), lisCount = this.DOMResults.getElementsByTagName("li").length;
                        if (position < 0) {
                            position = lisCount - 1;
                        }
                        else if (position >= lisCount) {
                            position = 0;
                        }
                        active.classList.remove("active");
                        active.parentElement.children.item(position).classList.add("active");
                    }
                    else if (last && event.keyCode == 38) {
                        last.classList.add("active");
                    }
                    else if (first) {
                        first.classList.add("active");
                    }
                },
                Operator: ConditionOperator.OR,
                Event: EventType.KEYUP
            },
            "AlphaNum": {
                Conditions: [{
                        Is: 13,
                        Not: true
                    }, {
                        From: 35,
                        To: 40,
                        Not: true
                    }],
                Callback: function () {
                    var oldValue = this.Input.getAttribute("data-autocomplete-old-value"), currentValue = this._Pre();
                    if (currentValue !== "" && currentValue.length >= this._MinChars()) {
                        if (!oldValue || currentValue != oldValue) {
                            this.DOMResults.setAttribute("class", "autocomplete open");
                        }
                        AutoComplete.prototype.cache(this, function (response) {
                            this._Render(this._Post(response));
                            this._Open();
                        }.bind(this), this._Error);
                    }
                    else {
                        this._Close();
                    }
                },
                Operator: ConditionOperator.AND,
                Event: EventType.KEYUP
            }
        },
        DOMResults: null,
        Request: null,
        Input: null,
        /**
         * Return the message when no result returns
         */
        _EmptyMessage: function () {
            var emptyMessage = "";
            if (this.Input.hasAttribute("data-autocomplete-empty-message")) {
                emptyMessage = this.Input.getAttribute("data-autocomplete-empty-message");
            }
            else if (this.EmptyMessage !== false) {
                emptyMessage = this.EmptyMessage;
            }
            else {
                emptyMessage = "";
            }
            return emptyMessage;
        },
        /**
         * Returns the maximum number of results
         */
        _Limit: function () {
            var limit = this.Input.getAttribute("data-autocomplete-limit");
            if (isNaN(limit) || limit === null) {
                return this.Limit;
            }
            return parseInt(limit, 10);
        },
        /**
         * Returns the minimum number of characters entered before firing ajax
         */
        _MinChars: function () {
            var minchars = this.Input.getAttribute("data-autocomplete-minchars");
            if (isNaN(minchars) || minchars === null) {
                return this.MinChars;
            }
            return parseInt(minchars, 10);
        },
        /**
         * Apply transformation on labels response
         */
        _Highlight: function (label) {
            return label.replace(this.Highlight.getRegex(this._Pre()), this.Highlight.transform);
        },
        /**
         * Returns the HHTP method to use
         */
        _HttpMethod: function () {
            if (this.Input.hasAttribute("data-autocomplete-method")) {
                return this.Input.getAttribute("data-autocomplete-method");
            }
            return this.HttpMethod;
        },
        /**
         * Returns the query param to use
         */
        _QueryArg: function () {
            if (this.Input.hasAttribute("data-autocomplete-param-name")) {
                return this.Input.getAttribute("data-autocomplete-param-name");
            }
            return this.QueryArg;
        },
        /**
         * Returns the URL to use for AJAX request
         */
        _Url: function () {
            if (this.Input.hasAttribute("data-autocomplete")) {
                return this.Input.getAttribute("data-autocomplete");
            }
            return this.Url;
        },
        /**
         * Manage the close
         */
        _Blur: function (now) {
            if (now === void 0) { now = false; }
            if (now) {
                this._Close();
            }
            else {
                var params = this;
                setTimeout(function () {
                    params._Blur(true);
                }, 150);
            }
        },
        /**
         * Manage the cache
         */
        _Cache: function (value) {
            return this.$Cache[value];
        },
        /**
         * Manage the open
         */
        _Focus: function () {
            var oldValue = this.Input.getAttribute("data-autocomplete-old-value");
            if ((!oldValue || this.Input.value != oldValue) && this._MinChars() <= this.Input.value.length) {
                this.DOMResults.setAttribute("class", "autocomplete open");
            }
        },
        /**
         * Bind all results item if one result is opened
         */
        _Open: function () {
            var params = this;
            Array.prototype.forEach.call(this.DOMResults.getElementsByTagName("li"), function (li) {
                if (li.getAttribute("class") != "locked") {
                    li.onclick = function () {
                        params._Select(li);
                    };
                }
            });
        },
        _Close: function () {
            this.DOMResults.setAttribute("class", "autocomplete");
        },
        /**
         * Position the results HTML element
         */
        _Position: function () {
            this.DOMResults.setAttribute("class", "autocomplete");
            this.DOMResults.setAttribute("style", "top:" + (this.Input.offsetTop + this.Input.offsetHeight) + "px;left:" + this.Input.offsetLeft + "px;width:" + this.Input.clientWidth + "px;");
        },
        /**
         * Execute the render of results DOM element
         */
        _Render: function (response) {
            var ul;
            if (typeof response == "string") {
                ul = this._RenderRaw(response);
            }
            else {
                ul = this._RenderResponseItems(response);
            }
            if (this.DOMResults.hasChildNodes()) {
                this.DOMResults.removeChild(this.DOMResults.childNodes[0]);
            }
            this.DOMResults.appendChild(ul);
        },
        /**
         * ResponseItems[] rendering
         */
        _RenderResponseItems: function (response) {
            var ul = document.createElement("ul"), li = document.createElement("li"), limit = this._Limit();
            // Order
            if (limit < 0) {
                response = response.reverse();
            }
            else if (limit === 0) {
                limit = response.length;
            }
            for (var item = 0; item < Math.min(Math.abs(limit), response.length); item++) {
                li.innerHTML = response[item].Label;
                li.setAttribute("data-autocomplete-value", response[item].Value);
                ul.appendChild(li);
                li = document.createElement("li");
            }
            return ul;
        },
        /**
         * string response rendering (RAW HTML)
         */
        _RenderRaw: function (response) {
            var ul = document.createElement("ul"), li = document.createElement("li");
            if (response.length > 0) {
                this.DOMResults.innerHTML = response;
            }
            else {
                var emptyMessage = this._EmptyMessage();
                if (emptyMessage !== "") {
                    li.innerHTML = emptyMessage;
                    li.setAttribute("class", "locked");
                    ul.appendChild(li);
                }
            }
            return ul;
        },
        /**
         * Deal with request response
         */
        _Post: function (response) {
            try {
                var returnResponse = [];
                //JSON return
                var json = JSON.parse(response);
                if (Object.keys(json).length === 0) {
                    return "";
                }
                if (Array.isArray(json)) {
                    for (var i = 0; i < Object.keys(json).length; i++) {
                        returnResponse[returnResponse.length] = { "Value": json[i], "Label": this._Highlight(json[i]) };
                    }
                }
                else {
                    for (var value in json) {
                        returnResponse.push({
                            "Value": value,
                            "Label": this._Highlight(json[value])
                        });
                    }
                }
                return returnResponse;
            }
            catch (event) {
                //HTML return
                return response;
            }
        },
        /**
         * Return the autocomplete value to send (before request)
         */
        _Pre: function () {
            return this.Input.value;
        },
        /**
         * Choice one result item
         */
        _Select: function (item) {
            if (item.hasAttribute("data-autocomplete-value")) {
                this.Input.value = item.getAttribute("data-autocomplete-value");
            }
            else {
                this.Input.value = item.innerHTML;
            }
            this.Input.setAttribute("data-autocomplete-old-value", this.Input.value);
        },
        /**
         * Handle HTTP error on the request
         */
        _Error: function () {
        },
        $AjaxTimer: null,
        $Cache: {},
        $Listeners: {}
    };
    return AutoComplete;
}());
module.exports = AutoComplete;

},{}]},{},[1])(1)
});
