window.onload = function(){
    var contactMe = new Vue({
        el: '#contact-me',

        data: {
            errors: [],
            name: null,
            email: null,
            reg: /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,24}))$/,
            phone: null,
            message: null,
            sent: false
        },

        methods: {
            submit: function (e) {
                if (this.name && this.email && this.phone && this.message) {                    
                    if(this.reg.test(this.email)){
                        this.sent = true;
                        return true;
                    }
                }
          
                this.errors = [];
          
                if (!this.name) this.errors.push('O nome é obrigatório.');
                if (!this.email) this.errors.push('O email é obrigatório.')
                else if(!this.reg.test(this.email)) this.errors.push('O email está incorreto!');
                if (!this.phone) this.errors.push('O telefone é obrigatório.');
                if (!this.message) this.errors.push('A mensagem é obrigatória.');
          
                e.preventDefault();
              }
        }
    })
}