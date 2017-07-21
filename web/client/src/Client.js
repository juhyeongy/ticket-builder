/* eslint-disable no-undef */
function predict({sourceId, text, linkTitle}, cb) {
  return fetch(`api/predict`, {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      sourceId,
      text,
      linkTitle
    })
  }).then(checkStatus)
    .then(parseJSON)
    .then(cb)
}

function predictTags({text}, cb) {
  return fetch(`api/predict-tags`, {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({text})
  }).then(checkStatus)
    .then(parseJSON)
    .then(cb)
}

function checkStatus(response) {
  if (response.status >= 200 && response.status < 300) {
    return response;
  }
  const error = new Error(`HTTP Error ${response.statusText}`);
  error.status = response.statusText;
  error.response = response;
  console.log(error); // eslint-disable-line no-console
  throw error;
}

function parseJSON(response) {
  return response.json();
}

const Client = {predict, predictTags};
export default Client;
