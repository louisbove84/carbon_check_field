// Vercel Serverless Function - Serve Google Maps API Key
// This keeps the API key secure by serving it from server-side
// URL: /api/carboncheck-config (Vercel strips .js from the route)

module.exports = function handler(req, res) {
  const apiKey = process.env.CARBONCHECK_GOOGLE_MAPS_API_KEY || '';

  res.setHeader('Content-Type', 'application/javascript');
  res.setHeader('Cache-Control', 'public, max-age=3600');

  res.status(200).send(`var GOOGLE_MAPS_API_KEY = '${apiKey}';`);
};
