
import React from 'react';

const MessageParser = ({ children, actions }) => {
    // console.log(children.props.state)
    const { checker } = children.props.state;
    const parse = (message) => {

        // if (checker === "process")
        if ((message.includes("process"))) 
            {
            actions.secondQuestion();
            children.props.state.userData.name = message;
        }

        // if (checker === "tax_balance")  
        if ((message.includes("balance")))
            {
            actions.thirdQuestion();
            children.props.state.userData.name = message;
        }

        // if (checker === "types") 
        if ((message.includes("types")))
            {
            actions.fourthQuestion();
            children.props.state.userData.name = message;
        }

        // if (checker === "deadline") 
        if ((message.includes("deadline")) && (message.includes("property")))
            {
            actions.fifthQuestion();
            children.props.state.userData.name = message;
        }

        // if (checker === "appeal") 
        if ((message.includes("assessment")))
            {
            actions.sixthQuestion();
            children.props.state.userData.name = message;
        }

        // if (checker === "exemptions") 
        if ((message.includes("exemptions")) || (message.includes("relief")))
            {
            actions.seventhQuestion();
            children.props.state.userData.name = message;
        }

        if ((message.includes("utilized"))) {
            actions.eigthQuestion();
            children.props.state.userData.name = message;
        }

        // if (checker === "preference" /*&& Number(message)*/) {
        //     actions.afterAgeMessage();
        //     children.props.state.userData.age = message;
        //     // if (message <= 10) {
        //     //     children.props.state.userData.category = "kid";
        //     // } else if (message > 10 && message <= 20) {
        //     //     children.props.state.userData.category = "teenagers";
        //     // } else {
        //     //     children.props.state.userData.category = "adults";
        //     // }
        // }
    }
    return (
        <div>
            {React.Children.map(children, (child) => {
                return React.cloneElement(child, {
                    parse: parse,
                    actions,
                });
            })}
        </div>
    );
};

export default MessageParser;
