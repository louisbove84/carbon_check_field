// Vercel Serverless Function - Serve Google Maps API Key
// This keeps the API key secure by serving it from server-side

export default function handler(req, res) {
  const apiKey = process.env.GOOGLE_MAPS_API_KEY || '';
  
  res.setHeader('Content-Type', 'application/javascript');
  res.setHeader('Cache-Control', 'public, max-age=3600');
  
  res.status(200).send(`var GOOGLE_MAPS_API_KEY = '${apiKey}';`);
}
