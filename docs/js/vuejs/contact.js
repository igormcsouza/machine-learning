window.onload = function(){
    var contactMe = new Vue({
        el: '#contact-me',

        data: {
            errors: [],
            name: null,
            email: null,
            phone: null,
            message: null,
            sent: false
        },

        methods: {
            submit: function (e) {
                if (this.name && this.email && this.phone && this.message) {
                    this.sent = true;
                    return true;
                }
          
                this.errors = [];
          
                if (!this.name) {
                    this.errors.push('O nome é obrigatório.');
                }
                if (!this.email) {
                    this.errors.push('O email é obrigatório.');
                }
                if (!this.phone) {
                    this.errors.push('O telefone é obrigatório.');
                }
                if (!this.message) {
                    this.errors.push('A mensagem é obrigatória.');
                }
          
                e.preventDefault();
              }
        }
    })
}