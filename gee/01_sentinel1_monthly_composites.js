
/***************************************************************
 SENTINEL-1 MONTHLY COMPOSITES - HEIDELBERG 2025

 Satellite: Sentinel-1 GRD
 Relative orbit: 15
 Pass direction: ASCENDING

 Period: January-December 2025

 Output resolution: 10 m

 Output bands:
   VV          - Monthly mean backscatter (dB)
   VH          - Monthly mean backscatter (dB)
   mean_n_obs  - Number of valid observations

 CRS: EPSG:32632

 Expected exports: 12 GeoTIFF files

 IMPORTANT:
 This script exports ONLY 10 m files.
 It does NOT create any 40 m export tasks.
***************************************************************/


// ============================================================
// 1. SETTINGS AND STUDY AREA
// ============================================================

var CRS = 'EPSG:32632';

var center = ee.Geometry.Point([8.675, 49.398])
  .transform(CRS, 1);

var AOI = center
  .buffer(20000)
  .bounds(1, CRS);

Map.centerObject(AOI, 9);

Map.addLayer(
  AOI,
  {color: 'red'},
  'Heidelberg study area'
);

print('Study area:', AOI);

print(
  'Study area km2:',
  AOI.area(1).divide(1000000)
);


// ============================================================
// 2. DEFINE EXACT 10 m OUTPUT GRID
// ============================================================

// Fixed grid in EPSG:32632.
// This is the same 10 m grid used in the previous script.

var GRID10 = ee.Projection(
  CRS,
  [10, 0, 0, 0, -10, 0]
);

print('10 m output projection:', GRID10);


// ============================================================
// 3. LOAD SENTINEL-1 GRD DATA
// ============================================================

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(AOI)
  .filterDate('2025-01-01', '2026-01-01')
  .filter(
    ee.Filter.eq('instrumentMode', 'IW')
  )
  .filter(
    ee.Filter.eq(
      'orbitProperties_pass',
      'ASCENDING'
    )
  )
  .filter(
    ee.Filter.eq(
      'relativeOrbitNumber_start',
      15
    )
  )
  .filter(
    ee.Filter.listContains(
      'transmitterReceiverPolarisation',
      'VV'
    )
  )
  .filter(
    ee.Filter.listContains(
      'transmitterReceiverPolarisation',
      'VH'
    )
  )
  .select(['VV', 'VH']);

print('Sentinel-1 total images:', s1.size());


// ============================================================
// 4. MONTHLY COMPOSITE FUNCTION
// ============================================================

function makeMonthlyComposite(month) {

  month = ee.Number(month);

  var start = ee.Date.fromYMD(2025, month, 1);

  var end = start.advance(1, 'month');

  // ----------------------------------------------------------
  // 4A. FILTER MONTHLY IMAGES
  // ----------------------------------------------------------

  var monthlyCollection = s1.filterDate(
    start,
    end
  );

  print(
    'Month image count:',
    month,
    monthlyCollection.size()
  );

  // ----------------------------------------------------------
  // 4B. APPLY COMMON VV AND VH MASK
  // ----------------------------------------------------------

  var commonMasked = monthlyCollection.map(
    function(image) {

      var vv = image.select('VV');

      var vh = image.select('VH');

      var commonMask = vv.mask().and(vh.mask());

      return image
        .select(['VV', 'VH'])
        .updateMask(commonMask)
        .setDefaultProjection(GRID10);

    }
  );

  // ----------------------------------------------------------
  // 4C. CONVERT dB TO LINEAR POWER
  // ----------------------------------------------------------

  // Linear power = 10^(dB / 10)

  var linearCollection = commonMasked.map(
    function(image) {

      return image
        .divide(10)
        .multiply(Math.LN10)
        .exp()
        .rename(['VV', 'VH'])
        .setDefaultProjection(GRID10);

    }
  );

  // ----------------------------------------------------------
  // 4D. MONTHLY TEMPORAL MEAN
  // ----------------------------------------------------------

  var meanPower = linearCollection
    .mean()
    .setDefaultProjection(GRID10);

  // ----------------------------------------------------------
  // 4E. CONVERT LINEAR POWER BACK TO dB
  // ----------------------------------------------------------

  var monthlyDB = meanPower
    .log10()
    .multiply(10)
    .rename(['VV', 'VH'])
    .toFloat()
    .setDefaultProjection(GRID10);

  // ----------------------------------------------------------
  // 4F. COUNT VALID OBSERVATIONS
  // ----------------------------------------------------------

  var observationCount = commonMasked
    .select('VV')
    .count()
    .rename('mean_n_obs')
    .toFloat()
    .setDefaultProjection(GRID10);

  // ----------------------------------------------------------
  // 4G. CREATE FINAL 10 m COMPOSITE
  // ----------------------------------------------------------

  // All bands are Float32.
  // This prevents the previous Float64/UInt32 export error.

  var composite10m = monthlyDB
    .addBands(observationCount)
    .toFloat()
    .setDefaultProjection(GRID10)
    .clip(AOI);

  return {
    month: month,
    imageCount: monthlyCollection.size(),
    image10m: composite10m
  };

}


// ============================================================
// 5. CREATE 12 MONTHLY 10 m EXPORT TASKS
// ============================================================

var monthNames = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec'
];

var exportFolder = 'Heidelberg_S1_2025_Orbit15';

for (var month = 1; month <= 12; month++) {

  var result = makeMonthlyComposite(month);

  var monthName = monthNames[month - 1];

  print(
    monthName + ' image count:',
    result.imageCount
  );

  // ----------------------------------------------------------
  // EXPORT ONLY 10 m
  // ----------------------------------------------------------

  Export.image.toDrive({

    image: result.image10m,

    description:
      'S1_2025_' + monthName + '_Orbit15_10m',

    folder: exportFolder,

    fileNamePrefix:
      'S1_2025_' + monthName + '_Orbit15_10m',

    region: AOI,

    crs: CRS,

    crsTransform: [
      10, 0, 0,
      0, -10, 0
    ],

    maxPixels: 1e13,

    fileFormat: 'GeoTIFF'

  });

}


// ============================================================
// 6. FINAL INFORMATION
// ============================================================

print('========================================');

print('SENTINEL-1 10 m EXPORT TASKS CREATED');

print('Year: 2025');

print('Relative orbit: 15');

print('Pass direction: ASCENDING');

print('Monthly composites: 12');

print('Output resolution: 10 m');

print('Bands: VV, VH, mean_n_obs');

print('Data type: Float32');

print('CRS: EPSG:32632');

print('Expected export tasks: 12');

print('Google Drive folder:', exportFolder);

print('========================================');
