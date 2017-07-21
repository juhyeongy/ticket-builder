'use strict'

const express = require('express');
const bodyParser = require('body-parser');
const request = require('request');
const app = express();

const host = 'localhost';
const port = '5000';

app.use(bodyParser.json());
app.set('port', (process.env.PORT || 3001));

if (process.env.NODE_ENV === 'production') {
  app.use(express.static('client/build'));
}

app.post('/api/predict-tags', (req, res) => {
  const text = req.body.text;

  let payload = {
    url: `http://${host}:${port}/api/preds/bugzilla-tags`,
    json: true,
    body: {
      text
    }
  };

  request.post(payload, (err, response, body) => {
    res.json(body);
  });
});

app.post('/api/predict', (req, res) => {
  const sourceId = req.body.sourceId;
  const text = req.body.text;
  const linkTitle = req.body.linkTitle;

  let payload = {
    url: `http://${host}:${port}/api/resources/${sourceId}/prediction`,
    json: true,
    body: {
      text,
      link_title: linkTitle
    }
  };

  request.post(payload, (err, response, body) => {
    res.json(body);
  });
});

app.listen(app.get('port'), () => {
  console.log(`Find the server at: http://localhost:${app.get('port')}/`); // eslint-disable-line no-console
});
