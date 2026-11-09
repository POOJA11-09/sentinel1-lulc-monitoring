
/***************************************************************
  HEIDELBERG DYNAMIC WORLD 2025
  9 ORIGINAL LAND-COVER CLASSES
  MONTHLY EXPORTS AT 10m AND 40m

  Dataset:
    GOOGLE/DYNAMICWORLD/V1

  Study area:
    Heidelberg, Germany
    Centre: 8.675, 49.398
    40 km x 40 km square
    EPSG:32632

  Outputs:
    12 monthly 10m GeoTIFFs
    12 monthly 40m GeoTIFFs

  10m bands:
    lulc
    confidence
    n_obs

  40m bands:
    lulc
    confidence
    n_obs
    purity

  Original Dynamic World class IDs:
    0 = Water
    1 = Trees
    2 = Grass
    3 = Flooded vegetation
    4 = Crops
    5 = Shrub and scrub
    6 = Built-up
    7 = Bare ground
    8 = Snow and ice

  All exported bands are Float32.
  Class codes remain exact integer values (0 to 8).
****************************************************************/



// =============================================================
// 1. STUDY AREA — CORRECTED GEOMETRY
// =============================================================

// Heidelberg centre point.
var center = ee.Geometry.Point([8.675, 49.398]);

// Target CRS.
var CRS = 'EPSG:32632';
var UTM = ee.Projection(CRS);

// Transform the centre point into UTM coordinates.
var centerUTM = center.transform(UTM, 1);

// Extract UTM coordinates.
var coords = centerUTM.coordinates();

var x = ee.Number(coords.get(0));
var y = ee.Number(coords.get(1));

// Create an exact 40 km x 40 km square.
// 20 km in each direction from the centre.
// No buffer() or bounds() operation is needed.

var aoi = ee.Geometry.Rectangle(
  [
    x.subtract(20000),
    y.subtract(20000),
    x.add(20000),
    y.add(20000)
  ],
  UTM,
  false
);

// Display study area.
Map.centerObject(aoi, 10);

Map.addLayer(
  aoi,
  {color: 'red'},
  'Heidelberg AOI'
);

// Print study area information.
print('Study area:', aoi);

print(
  'AOI area in km2:',
  aoi.area(1, UTM).divide(1e6)
);



// =============================================================
// 2. SETTINGS
// =============================================================

var YEAR = 2025;

// Common aligned grids.
// Both grids have the same origin (0, 0).
// One 40m pixel covers a 4 x 4 block of 10m pixels.

var TRANSFORM10 = [10, 0, 0, 0, -10, 0];
var TRANSFORM40 = [40, 0, 0, 0, -40, 0];

var GRID10 = ee.Projection(CRS, TRANSFORM10);

// Dynamic World probability bands in original class order.
var CLASS_BANDS = [
  'water',
  'trees',
  'grass',
  'flooded_vegetation',
  'crops',
  'shrub_and_scrub',
  'built',
  'bare',
  'snow_and_ice'
];

// Original class names.
var CLASS_NAMES = [
  'Water',
  'Trees',
  'Grass',
  'Flooded vegetation',
  'Crops',
  'Shrub and scrub',
  'Built-up',
  'Bare ground',
  'Snow and ice'
];

// Dynamic World original class colours.
var PALETTE = [
  '419bdf',
  '397d49',
  '88b053',
  '7a87c6',
  'e49635',
  'dfc35a',
  'c4281b',
  'a59b8f',
  'b39fe1'
];

// Google Drive output folder.
var DRIVE_FOLDER = 'Heidelberg_DW_2025_9class';


// =============================================================
// 3. LOAD DYNAMIC WORLD 2025
// =============================================================

var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterBounds(aoi)
  .filterDate(
    ee.Date.fromYMD(YEAR, 1, 1),
    ee.Date.fromYMD(YEAR + 1, 1, 1)
  )
  .select(CLASS_BANDS);

print('Dynamic World collection:', dw);
print('Total Dynamic World images in 2025:', dw.size());


// =============================================================
// 4. MONTHLY LABEL CREATION FUNCTION
// =============================================================

function createMonthlyLabels(month) {

  var start = ee.Date.fromYMD(YEAR, month, 1);
  var end = start.advance(1, 'month');

  // Select observations for this month.
  var monthly = dw.filterDate(start, end);

  // Monthly image count.
  var monthlyCount = monthly.size();

  // -----------------------------------------------------------
  // 4A. MONTHLY MEAN CLASS PROBABILITIES
  // -----------------------------------------------------------

  // Average the nine probability bands independently.
  var meanProb = monthly
    .mean()
    .setDefaultProjection(GRID10);

  // Count valid observations per pixel.
  var nObs10 = monthly
    .select('water')
    .count()
    .rename('n_obs')
    .setDefaultProjection(GRID10);

  // -----------------------------------------------------------
  // 4B. 10m LAND-COVER LABEL
  // -----------------------------------------------------------

  // Find the position of the highest mean probability.
  // The band order matches original Dynamic World IDs 0-8.

  var lulc10 = meanProb
    .toArray()
    .arrayArgmax()
    .arrayGet([0])
    .rename('lulc')
    .setDefaultProjection(GRID10);

  // Maximum mean probability across all nine classes.
  var confidence10 = meanProb
    .reduce(ee.Reducer.max())
    .rename('confidence')
    .setDefaultProjection(GRID10);

  // Combine 10m bands.
  // All bands stored as Float32.
  var image10 = ee.Image.cat([
    lulc10,
    confidence10,
    nObs10
  ]).toFloat();

  // -----------------------------------------------------------
  // 4C. CREATE NINE ONE-HOT CLASS BANDS
  // -----------------------------------------------------------

  // Each 10m pixel receives:
  // 1 for its predicted class.
  // 0 for the other eight classes.

  var oneHotBands = [];

  for (var classId = 0; classId < 9; classId++) {

    var classBand = lulc10
      .eq(classId)
      .rename('class_' + classId)
      .toFloat();

    oneHotBands.push(classBand);
  }

  var oneHot10 = ee.Image.cat(oneHotBands)
    .setDefaultProjection(GRID10);

  // -----------------------------------------------------------
  // 4D. AGGREGATE CLASS PROPORTIONS TO 40m
  // -----------------------------------------------------------

  // Average the one-hot class bands over each 40m pixel.
  // This gives the fraction of valid 10m pixels belonging
  // to each class.

  var classFractions40 = oneHot10
    .reduceResolution({
      reducer: ee.Reducer.mean(),
      maxPixels: 64
    })
    .reproject({
      crs: CRS,
      crsTransform: TRANSFORM40
    });

  // -----------------------------------------------------------
  // 4E. 40m MAJORITY CLASS
  // -----------------------------------------------------------

  // Select the class with the highest fraction.
  // This is the majority class for each 40m pixel.

  var lulc40 = classFractions40
    .toArray()
    .arrayArgmax()
    .arrayGet([0])
    .rename('lulc');

  // -----------------------------------------------------------
  // 4F. 40m CLASS PURITY
  // -----------------------------------------------------------

  // Fraction of valid 10m pixels belonging to the dominant
  // class within each 40m pixel.

  var purity40 = classFractions40
    .reduce(ee.Reducer.max())
    .rename('purity');

  // -----------------------------------------------------------
  // 4G. 40m MEAN CONFIDENCE
  // -----------------------------------------------------------

  var confidence40 = confidence10
    .reduceResolution({
      reducer: ee.Reducer.mean(),
      maxPixels: 64
    })
    .reproject({
      crs: CRS,
      crsTransform: TRANSFORM40
    })
    .rename('confidence');

  // -----------------------------------------------------------
  // 4H. 40m MEAN OBSERVATION COUNT
  // -----------------------------------------------------------

  var nObs40 = nObs10
    .reduceResolution({
      reducer: ee.Reducer.mean(),
      maxPixels: 64
    })
    .reproject({
      crs: CRS,
      crsTransform: TRANSFORM40
    })
    .rename('n_obs');

  // Combine 40m bands.
  // All bands stored as Float32.

  var image40 = ee.Image.cat([
    lulc40,
    confidence40,
    nObs40,
    purity40
  ]).toFloat();

  return {
    image10: image10,
    image40: image40,
    monthlyCount: monthlyCount
  };
}


// =============================================================
// 5. CREATE 24 EXPORT TASKS
// =============================================================

for (var month = 1; month <= 12; month++) {

  var result = createMonthlyLabels(month);

  var monthText = (month < 10 ? '0' : '') + month;

  print(
    'Dynamic World images for month ' + monthText,
    result.monthlyCount
  );

  // -----------------------------------------------------------
  // 5A. EXPORT 10m
  // -----------------------------------------------------------

  Export.image.toDrive({

    image: result.image10,

    description:
      'DW_2025_' + monthText + '_10m_9class',

    folder: DRIVE_FOLDER,

    fileNamePrefix:
      'DW_2025_' + monthText + '_10m_9class',

    region: aoi,

    crs: CRS,

    crsTransform: TRANSFORM10,

    maxPixels: 1e13,

    fileFormat: 'GeoTIFF'

  });


  // -----------------------------------------------------------
  // 5B. EXPORT 40m
  // -----------------------------------------------------------

  Export.image.toDrive({

    image: result.image40,

    description:
      'DW_2025_' + monthText + '_40m_9class',

    folder: DRIVE_FOLDER,

    fileNamePrefix:
      'DW_2025_' + monthText + '_40m_9class',

    region: aoi,

    crs: CRS,

    crsTransform: TRANSFORM40,

    maxPixels: 1e13,

    fileFormat: 'GeoTIFF'

  });

}


// =============================================================
// 6. DISPLAY JANUARY 2025 MAP
// =============================================================

var january = createMonthlyLabels(1);

// Display 10m land-cover classes.
Map.addLayer(
  january.image10.select('lulc'),
  {
    min: 0,
    max: 8,
    palette: PALETTE
  },
  'Dynamic World January 2025 - 9 classes'
);

// Display confidence, hidden by default.
Map.addLayer(
  january.image10.select('confidence'),
  {
    min: 0,
    max: 1,
    palette: ['red', 'yellow', 'green']
  },
  'January confidence',
  false
);

// Display 40m purity, hidden by default.
Map.addLayer(
  january.image40.select('purity'),
  {
    min: 0,
    max: 1,
    palette: ['red', 'yellow', 'green']
  },
  'January 40m class purity',
  false
);


// =============================================================
// 7. PRINT CLASS INFORMATION AND EXPORT SUMMARY
// =============================================================

print('Original Dynamic World class IDs:', ee.List.sequence(0, 8));

print('Original Dynamic World class names:', CLASS_NAMES);

print('Output Google Drive folder:', DRIVE_FOLDER);

print('10m grid transform:', TRANSFORM10);

print('40m grid transform:', TRANSFORM40);

print('Expected number of export tasks:', 24);

print('All 24 export tasks have been created.');

print('Open the Tasks tab to start the exports.');
