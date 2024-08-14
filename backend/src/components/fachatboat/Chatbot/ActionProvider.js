import React from 'react';

const ActionProvider = ({ createChatBotMessage, setState, children }) => {

    const initialAction = () => {
        const message = createChatBotMessage('What is your problem ?');
        updateState(message)
    }

    const secondQuestion = () => {
        const message = createChatBotMessage("You can pay your municipal taxes online by visiting our official website. Navigate to the 'Online Payments' section, select 'Tax Payment,' and follow the instructions to complete your payment securely.")
        updateState(message)
    }

    const thirdQuestion = () => {
        const message = createChatBotMessage('To check your outstanding tax balance, log in to your account on our website and go to the "My Account" section. Here, you will find detailed information about your current and past dues.')
        updateState(message)
    }

    const fourthQuestion = () => {
        const message = createChatBotMessage('The municipal corporation collects various types of taxes, including property tax, water tax, and sanitation tax. Each tax helps fund essential public services and infrastructure maintenance.')
        updateState(message)
    }

    const fifthQuestion = () => {
        const message = createChatBotMessage('Property tax payments are due annually by March 31st. Late payments may incur additional penalties and interest. Please refer to the "Tax Deadlines" section on our website for more information.')
        updateState(message)
    }

    const sixthQuestion = () => {
        const message = createChatBotMessage(' If you believe your property tax assessment is incorrect, you can file an appeal by submitting the necessary documentation through the "Tax Appeals" section on our website. Our team will review your case and get back to you with a resolution.')
        updateState(message)
    }

    const seventhQuestion = () => {
        const message = createChatBotMessage('Yes, there are several tax exemptions and relief programs available for eligible residents, including senior citizens, veterans, and low-income households. Visit the "Tax Exemptions" section on our website for eligibility criteria and application procedures.')
        updateState(message)
    }

    const eigthQuestion = () => {
        const message = createChatBotMessage('Collected tax revenue is used to fund essential public services such as waste management, road maintenance, water supply, and public safety. Detailed reports on revenue utilization can be found in the "Revenue Reports" section of our website.')
        updateState(message)
    }

    // const afterNameMessage = () => {
    //     const message = createChatBotMessage("You can pay your municipal taxes online by visiting our official website. Navigate to the 'Online Payments' section, select 'Tax Payment,' and follow the instructions to complete your payment securely.")
    //     updateState(message, "age")
    // }

    // const afterAgeMessage = () => {
    //     const message = createChatBotMessage('To check your outstanding tax balance, log in to your account on our website and go to the "My Account" section. Here, you will find detailed information about your current and past dues.'
    //         // , {widget: "startSlow"}
    // )
    //     updateState(message)
    // }

    // const finalResult = (name, age, preference, vehicle) => {
    //     const message = createChatBotMessage(`Got it, ${name}! Based on your age ${age} and preference for a ${preference} ride, I recommend the '${vehicle}.' Enjoy the thrill!`, {
    //         widget: "finalImage"
    //     })
    //     updateState(message)
    // }

    const updateState = (message, checker) => {
        setState((prev) => ({
            ...prev,
            messages: [...prev.messages, message],
            checker,
        }))
    }

    return (
        <div>
            {React.Children.map(children, (child) => {
                return React.cloneElement(child, {
                    actions: {
                        initialAction,
                        secondQuestion,
                        thirdQuestion,
                        fourthQuestion,
                        fifthQuestion,
                        sixthQuestion,
                        seventhQuestion,
                        eigthQuestion
                        // afterNameMessage,
                        // afterAgeMessage,
                        // finalResult
                    },
                });
            })}
        </div>
    );
};

export default ActionProvider;