module.exports = (req, res) => {
  const mapsKey = process.env.CARBONCHECK_GOOGLE_MAPS_API_KEY || '';
  res.setHeader('Content-Type', 'application/javascript; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.status(200).send(`var GOOGLE_MAPS_API_KEY = '${mapsKey}';`);
};
