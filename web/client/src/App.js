import React, { Component } from 'react';
import NewArticle from './NewArticle';
import classNames from 'classnames';
import Client from './Client';
import _ from 'lodash';

class App extends Component {
  state = {
    scores: {},
    tags: {}
  }

  constructor(props) {
    super(props);
    this.fetchScores = this.fetchScores.bind(this);
    this.fetchTags = this.fetchTags.bind(this);
    this.resetTags = this.resetTags.bind(this);
  }

  fetchScores({text, linkTitle}) {
    const sourceId = this.props.sourceId;
    Client.predict({sourceId, text, linkTitle}, (scores) => {
      this.setState({
        scores: scores
      })
    });
  }

  fetchTags({text}) {
    Client.predictTags({text}, (tags) => {
      this.setState({
        tags: tags
      })
    });
  }

  resetTags() {
    this.setState({
      tags: {}
    })
  }

  render() {
    var btnClass = classNames('ui blue basic button');
    const tags = this.state.tags;
    const tagRows = _.map(tags, (label, name) => (
      <div className="extra content" key={label}>
        <div className="ui blue labeled button">
          <div className="ui button">
            {name}
          </div>
          <div className="ui basic label">
            {label}
          </div>
        </div>
      </div>
    ));

    return (
      <div className="App">
        <div className="ui container">

          <div className="ui grid centered">
            <div className="row app-title">
              <h1>
                Bugzilla Tag Recommendation Engine
              </h1>
            </div>
            <div className="ui divider"></div>
            <div className="row">
              <div className="ten wide column">
                <NewArticle
                  onSubmit={this.fetchTags}
                  onClear={this.resetTags}
                />
              </div>
              <div className="six wide column">
                <div className="ui card">
                  <div className="content">
                    <div className="header">Suggested Tags</div>
                    <div className="meta">Based on model trained on bugzilla tickets</div>
                  </div>
                  {tagRows}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }
}

export default App;
