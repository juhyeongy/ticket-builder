
import React from 'react';
import Client from './Client';
import classNames from 'classnames';

class NewArticle extends React.Component {
  state = {
    text: ''
  };

  constructor(props) {
    super(props);
    this.handleTextChange = this.handleTextChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
    this.handleClick = this.handleClick.bind(this);
    this.handleClear = this.handleClear.bind(this);
  }

  handleTextChange = (e) => {
    const value = e.target.value;
    this.setState({ text: value });
  };

  handleClick = (e) => {
    this.setState({ text: '' }, () => {
      this.props.onClick();
    })
  };

  handleSubmit = (e) => {
    this.props.onSubmit(this.state);
  };

  handleClear = (e) => {
    this.setState({ text: '' }, () => {
      this.refs.emailTextarea.value = ''
      this.props.onClear(this.state);
    })
  };

  render() {
    return (
      <div className="ui orange segment">
        <div className="ui comments">
          <div className="comment">
            <div className="content">
              <div className="text">
                <h4>
                  Customer Inquiry Form
                </h4>
              </div>
              <form className="ui reply form">
                <div className="field">
                  <textarea
                    ref="emailTextarea"
                    placeholder="Paste customer's inquiry ticket here"
                    name="text"
                    onChange={this.handleTextChange}
                  />
                </div>
                <div
                  className="ui primary submit labeled icon button"
                  onClick={this.handleSubmit}
                >
                  <i className="icon edit"></i> Populate Bugzilla Tags
                </div>

                <div
                  className="ui primary labeled icon button"
                  onClick={this.handleClear}
                >
                  <i className="icon cancel"></i> Clear
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>
    )
  }
}

export default NewArticle;
